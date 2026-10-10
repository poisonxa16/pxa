/* pxa-update — check and apply a PXA release.
 * The GUI only launches this. Download and checksum stay here, not in Python.
 * curl and sha256sum are the same tools install.sh already uses.
 *
 * The version directory that `current` points at is only the release.
 * Models, settings, and *.expert-counts.csv stay outside it. This tool
 * never puts them there, and it refuses an archive that contains them.
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static void die(int code, const char *msg)
{
    fprintf(stderr, "pxa-update: %s\n", msg);
    exit(code);
}

static int run_capture(char *const argv[], char *out, size_t outlen)
{
    int pipefd[2];
    if (pipe(pipefd) != 0)
        return -1;
    pid_t p = fork();
    if (p < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (p == 0) {
        dup2(pipefd[1], 1);
        close(pipefd[0]);
        close(pipefd[1]);
        execvp(argv[0], argv);
        _exit(127);
    }
    close(pipefd[1]);
    size_t i = 0;
    char c;
    while (read(pipefd[0], &c, 1) == 1) {
        if (out && i + 1 < outlen)
            out[i++] = c;
    }
    if (out)
        out[i] = 0;
    close(pipefd[0]);
    int st = 0;
    if (waitpid(p, &st, 0) < 0)
        return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

static int run(char *const argv[])
{
    return run_capture(argv, NULL, 0);
}

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0 || n > 8 * 1024 * 1024) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char *b = malloc((size_t)n + 1);
    if (!b) {
        fclose(f);
        return NULL;
    }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        fclose(f);
        return NULL;
    }
    b[n] = 0;
    if (len)
        *len = (size_t)n;
    fclose(f);
    return b;
}

/* 0 and http set to the status. Non-zero on a transport error or a non-200. */
static int fetch(const char *url, const char *dest, int *http)
{
    char code[16] = {0};
    char *argv[] = {"curl", "-sS", "--retry", "2", "--connect-timeout", "15", "--max-time", "1800",
                    "-o", (char *)dest, "-w", "%{http_code}", (char *)url, NULL};
    int rc = run_capture(argv, code, sizeof code);
    int status = atoi(code);
    if (http)
        *http = status;
    if (rc != 0 || status != 200)
        return -1;
    return 0;
}

static int http_ok(const char *url)
{
    return strncmp(url, "https://", 8) == 0 || strncmp(url, "http://127.0.0.1:", 16) == 0
        || strncmp(url, "http://localhost:", 17) == 0;
}

static int ends_with(const char *s, const char *suf)
{
    size_t n = strlen(s), m = strlen(suf);
    return n >= m && memcmp(s + (n - m), suf, m) == 0;
}

static const char *base_name(const char *path)
{
    const char *b = strrchr(path, '/');
    return b ? b + 1 : path;
}

/* numeric tags: v3 < v3.1 < v3.1.1. A v2026.10 date tag is the old line, older than v3.
 * A trailing -rc suffix is older than the same numbers with nothing after it. */
static int vercmp(const char *a, const char *b)
{
    if (a[0] == 'v' || a[0] == 'V')
        a++;
    if (b[0] == 'v' || b[0] == 'V')
        b++;
    char *ea = NULL, *eb = NULL;
    long fa = strtol(a, &ea, 10);
    long fb = strtol(b, &eb, 10);
    int ca = (ea != a && fa >= 2000);
    int cb = (eb != b && fb >= 2000);
    if (ca != cb)
        return ca ? -1 : 1;
    for (;;) {
        int a_ok = isdigit((unsigned char)*a);
        int b_ok = isdigit((unsigned char)*b);
        if (!a_ok && !b_ok) {
            if (*a && !*b)
                return -1;
            if (*b && !*a)
                return 1;
            return 0;
        }
        ea = NULL;
        eb = NULL;
        long xa = a_ok ? strtol(a, &ea, 10) : 0;
        long xb = b_ok ? strtol(b, &eb, 10) : 0;
        if (xa != xb)
            return xa < xb ? -1 : 1;
        if (a_ok) {
            a = ea;
            if (*a == '.')
                a++;
        }
        if (b_ok) {
            b = eb;
            if (*b == '.')
                b++;
        }
    }
}

/* 1 if the system glibc is new enough, 0 if it is older, -1 if we cannot tell.
 * Unknown must not be treated as new: that used to pick the Ubuntu 24.04 build. */
static int glibc_at_least(int maj, int min)
{
    char buf[64] = {0};
    FILE *f = popen("getconf GNU_LIBC_VERSION", "r");
    if (!f)
        return -1;
    if (!fgets(buf, sizeof buf, f)) {
        pclose(f);
        return -1;
    }
    int st = pclose(f);
    if (st != 0)
        return -1;
    int a = 0, b = 0;
    if (sscanf(buf, "glibc %d.%d", &a, &b) != 2 && sscanf(buf, "%d.%d", &a, &b) != 2)
        return -1;
    return a > maj || (a == maj && b >= min);
}

static const char *json_str_after(const char *hay, const char *key, char *out, size_t outlen)
{
    const char *p = strstr(hay, key);
    if (!p)
        return NULL;
    p = strchr(p, ':');
    if (!p)
        return NULL;
    p = strchr(p, '"');
    if (!p)
        return NULL;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < outlen) {
        if (*p == '\\' && p[1])
            p++;
        out[i++] = *p++;
    }
    out[i] = 0;
    return out;
}

/* Exact engine names from the release notes. A .sha256 sidecar contains ".tar.gz"
 * in the middle and must not be chosen. */
static int is_u22(const char *base)
{
    return strncmp(base, "pxa-", 4) == 0 && strstr(base, "linux-x86_64")
        && ends_with(base, "-ubuntu22.04.tar.gz");
}

static int is_u24(const char *base)
{
    if (strncmp(base, "pxa-", 4) != 0 || !strstr(base, "linux-x86_64"))
        return 0;
    if (strstr(base, ".sha256") || strstr(base, "-ubuntu22.04"))
        return 0;
    return ends_with(base, "-sm60_61_70.tar.gz") || ends_with(base, "-ubuntu24.04.tar.gz");
}

/* 0 ok, -1 no tarball, -2 glibc unknown. */
static int pick_url(const char *json, char *url, size_t urllen)
{
    const char *p = json;
    char u24[1024] = {0};
    char u22[1024] = {0};
    while ((p = strstr(p, "browser_download_url")) != NULL) {
        char one[1024];
        if (!json_str_after(p, "browser_download_url", one, sizeof one))
            break;
        p += 20;
        const char *base = base_name(one);
        if (strstr(base, "libggml-pxqn"))
            continue;
        if (is_u22(base)) {
            if (!u22[0])
                snprintf(u22, sizeof u22, "%s", one);
        } else if (is_u24(base)) {
            if (!u24[0])
                snprintf(u24, sizeof u24, "%s", one);
        }
    }
    int glibc = glibc_at_least(2, 38);
    if (glibc < 0)
        return -2;
    const char *pick = glibc ? (u24[0] ? u24 : u22) : u22;
    if (!pick || !pick[0])
        return -1;
    if (!http_ok(pick))
        return -1;
    snprintf(url, urllen, "%s", pick);
    return 0;
}

/* VERSION is either a bare tag ("v3") or the release file ("tag:              v3"). */
static void version_token(const char *path, char *out, size_t outlen)
{
    char *b = slurp(path, NULL);
    if (!b) {
        snprintf(out, outlen, "0");
        return;
    }
    const char *p = b;
    if (strncmp(p, "tag:", 4) == 0) {
        p += 4;
        while (*p == ' ' || *p == '\t')
            p++;
    }
    size_t i = 0;
    while (p[i] && p[i] != '\n' && p[i] != '\r' && p[i] != ' ' && p[i] != '\t' && i + 1 < outlen) {
        out[i] = p[i];
        i++;
    }
    out[i] = 0;
    free(b);
    if (!out[0])
        snprintf(out, outlen, "0");
}

static int read_version(const char *dir, char *out, size_t outlen)
{
    char path[PATH_MAX];
    snprintf(path, sizeof path, "%s/current/VERSION", dir);
    if (access(path, R_OK) == 0) {
        version_token(path, out, outlen);
        return 0;
    }
    snprintf(path, sizeof path, "%s/VERSION", dir);
    if (access(path, R_OK) == 0) {
        version_token(path, out, outlen);
        return 0;
    }
    snprintf(path, sizeof path, "%s/current", dir);
    char real[PATH_MAX];
    if (realpath(path, real)) {
        const char *base = strrchr(real, '/');
        base = base ? base + 1 : real;
        if (strncmp(base, "pxa-", 4) == 0)
            base += 4;
        snprintf(out, outlen, "%s", base[0] ? base : "0");
        return 0;
    }
    snprintf(out, outlen, "0");
    return 0;
}

/* The unpacked package that contains this binary: .../pxa-v3/bin/pxa-update and .../pxa-v3/VERSION. */
static int own_package(char *dir, size_t dirlen)
{
    char exe[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n < 0)
        return -1;
    exe[n] = 0;
    for (int i = 0; i < 5; i++) {
        char *sl = strrchr(exe, '/');
        if (!sl || sl == exe)
            return -1;
        *sl = 0;
        char v[PATH_MAX];
        snprintf(v, sizeof v, "%s/VERSION", exe);
        if (access(v, R_OK) == 0) {
            snprintf(dir, dirlen, "%s", exe);
            return 0;
        }
    }
    return -1;
}

static int server_running(const char *dir)
{
    char root[PATH_MAX];
    if (!realpath(dir, root))
        return 0;
    size_t nroot = strlen(root);
    DIR *d = opendir("/proc");
    if (!d)
        return 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!isdigit((unsigned char)ent->d_name[0]))
            continue;
        char link[64], exe[PATH_MAX];
        snprintf(link, sizeof link, "/proc/%s/exe", ent->d_name);
        ssize_t n = readlink(link, exe, sizeof exe - 1);
        if (n < 0)
            continue;
        exe[n] = 0;
        if (strncmp(exe, root, nroot) != 0)
            continue;
        if (exe[nroot] != '/' && exe[nroot] != 0)
            continue;
        if (strstr(exe, "llama-server")) {
            closedir(d);
            return 1;
        }
    }
    closedir(d);
    return 0;
}

static int sha_ok(const char *tgz, const char *sidecar)
{
    char *side = slurp(sidecar, NULL);
    if (!side)
        return 0;
    char want[80] = {0};
    if (sscanf(side, "%64s", want) != 1) {
        free(side);
        return 0;
    }
    free(side);
    if (strlen(want) != 64)
        return 0;
    int pipefd[2];
    if (pipe(pipefd) != 0)
        return 0;
    pid_t p = fork();
    if (p < 0)
        return 0;
    if (p == 0) {
        dup2(pipefd[1], 1);
        close(pipefd[0]);
        close(pipefd[1]);
        execlp("sha256sum", "sha256sum", tgz, (char *)NULL);
        _exit(127);
    }
    close(pipefd[1]);
    char got[80] = {0};
    size_t i = 0;
    char c;
    while (i < 64 && read(pipefd[0], &c, 1) == 1 && c != ' ' && c != '\n')
        got[i++] = c;
    close(pipefd[0]);
    int st = 0;
    waitpid(p, &st, 0);
    return i == 64 && strcmp(want, got) == 0;
}

static int path_component(const char *path, const char *name)
{
    const char *p = strchr(path, '/');
    if (!p)
        return strcmp(path, name) == 0;
    p++;
    size_t nlen = strlen(name);
    while (*p) {
        size_t n = strcspn(p, "/");
        if (n == nlen && strncmp(p, name, n) == 0)
            return 1;
        p += n;
        if (*p == '/')
            p++;
    }
    return 0;
}

/* User data is never part of a release. Models, settings, and expert-count files
 * live beside the version directory, not inside the tree `current` points at. */
static int user_payload(const char *path)
{
    if (strstr(path, "..") || path[0] == '/')
        return 1;
    if (path_component(path, "models") || path_component(path, "configs"))
        return 1;
    const char *base = base_name(path);
    if (strcmp(base, "control.json") == 0)
        return 1;
    if (ends_with(base, ".expert-counts.csv"))
        return 1;
    return 0;
}

static void rm_tree(const char *path)
{
    char *argv[] = {"rm", "-rf", "--", (char *)path, NULL};
    run(argv);
}

/* Read the archive. Fills top with the single first path component.
 * Returns -1 on a bad archive, -2 if it carries user files. */
static int archive_top(const char *tgz, char *top, size_t toplen)
{
    int pipefd[2];
    if (pipe(pipefd) != 0)
        return -1;
    pid_t p = fork();
    if (p < 0)
        return -1;
    if (p == 0) {
        dup2(pipefd[1], 1);
        close(pipefd[0]);
        close(pipefd[1]);
        execlp("tar", "tar", "-tzf", tgz, (char *)NULL);
        _exit(127);
    }
    close(pipefd[1]);
    char line[PATH_MAX];
    size_t k = 0;
    int saw = 0;
    int bad = 0;
    char c;
    top[0] = 0;
    while (read(pipefd[0], &c, 1) == 1) {
        if (c != '\n') {
            if (k + 1 < sizeof line)
                line[k++] = c;
            continue;
        }
        line[k] = 0;
        k = 0;
        if (!line[0])
            continue;
        if (user_payload(line))
            bad = 1;
        if (!saw) {
            char *slash = strchr(line, '/');
            if (slash)
                *slash = 0;
            if (!line[0] || strcmp(line, ".") == 0 || strcmp(line, "..") == 0 || strchr(line, '/'))
                bad = 1;
            else
                snprintf(top, toplen, "%s", line);
            saw = 1;
        }
    }
    close(pipefd[0]);
    int st = 0;
    waitpid(p, &st, 0);
    if (!saw || !top[0] || bad)
        return bad ? -2 : -1;
    return 0;
}

static int rel_link(const char *path, char *out, size_t outlen)
{
    ssize_t n = readlink(path, out, outlen - 1);
    if (n < 0)
        return -1;
    out[n] = 0;
    if (!out[0] || strchr(out, '/'))
        return -1;
    return 0;
}

static int swap_link(const char *dir, const char *name, const char *target)
{
    char tmp[PATH_MAX], dest[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s/%s.new", dir, name);
    snprintf(dest, sizeof dest, "%s/%s", dir, name);
    unlink(tmp);
    if (symlink(target, tmp) != 0)
        return -1;
    if (rename(tmp, dest) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

static int apply_release(const char *dir, const char *url)
{
    if (server_running(dir))
        die(2, "a server from this install is running; stop it first");
    if (mkdir(dir, 0755) != 0 && errno != EEXIST)
        die(1, "could not create the install directory");
    const char *base = base_name(url);
    if (!ends_with(base, ".tar.gz") || strstr(base, ".sha256"))
        die(1, "that is not a release tarball");
    /* /tmp can be small or noexec. The tarball stays on the install filesystem. */
    char dl[PATH_MAX];
    if (snprintf(dl, sizeof dl, "%s/.pxa-update-dl-XXXXXX", dir) >= (int) sizeof dl)
        die(1, "install path is too long");
    if (!mkdtemp(dl))
        die(1, "could not make a download directory");
    char tgz[PATH_MAX], side[PATH_MAX];
    snprintf(tgz, sizeof tgz, "%s/%s", dl, base);
    snprintf(side, sizeof side, "%s.sha256", tgz);
    char surl[1200];
    snprintf(surl, sizeof surl, "%s.sha256", url);
    if (!http_ok(url) || !http_ok(surl)) {
        rm_tree(dl);
        die(1, "refusing that download address");
    }
    int http = 0;
    if (fetch(url, tgz, &http) != 0) {
        rm_tree(dl);
        if (http == 403)
            die(1, "GitHub rate limit. Wait a minute and try again.");
        die(1, "download failed");
    }
    if (fetch(surl, side, &http) != 0) {
        rm_tree(dl);
        die(1, "checksum download failed");
    }
    if (!sha_ok(tgz, side)) {
        rm_tree(dl);
        die(1, "checksum mismatch; not installing");
    }
    char top[PATH_MAX];
    int arc = archive_top(tgz, top, sizeof top);
    if (arc == -2) {
        rm_tree(dl);
        die(1, "this archive contains models, settings, or expert counts; refusing");
    }
    if (arc != 0) {
        rm_tree(dl);
        die(1, "archive has no top directory");
    }
    char dest[PATH_MAX], ver[PATH_MAX];
    snprintf(dest, sizeof dest, "%s/%s", dir, top);
    snprintf(ver, sizeof ver, "%s/VERSION", dest);
    if (access(ver, R_OK) != 0) {
        /* Staging stays on the install filesystem so the move is atomic. */
        char stage_in[PATH_MAX];
        snprintf(stage_in, sizeof stage_in, "%s/.pxa-stage-XXXXXX", dir);
        if (!mkdtemp(stage_in)) {
            rm_tree(dl);
            die(1, "could not make a staging directory");
        }
        char *argv[] = {"tar", "-xzf", tgz, "-C", stage_in, NULL};
        if (run(argv) != 0) {
            rm_tree(stage_in);
            rm_tree(dl);
            die(1, "could not unpack");
        }
        char unpacked[PATH_MAX];
        snprintf(unpacked, sizeof unpacked, "%s/%s", stage_in, top);
        if (rename(unpacked, dest) != 0) {
            rm_tree(stage_in);
            rm_tree(dl);
            die(1, "could not move the new version into place");
        }
        rmdir(stage_in);
    }
    char curlink[PATH_MAX];
    snprintf(curlink, sizeof curlink, "%s/current", dir);
    char old[PATH_MAX];
    if (rel_link(curlink, old, sizeof old) == 0 && strcmp(old, top) != 0)
        swap_link(dir, "previous", old);
    if (swap_link(dir, "current", top) != 0) {
        rm_tree(dl);
        die(1, "could not switch current");
    }
    rm_tree(dl);
    printf("installed %s\n", top);
    printf("next %s/current\n", dir);
    return 0;
}

static int rollback(const char *dir)
{
    if (server_running(dir))
        die(2, "a server from this install is running; stop it first");
    char prevpath[PATH_MAX];
    snprintf(prevpath, sizeof prevpath, "%s/previous", dir);
    char prev[PATH_MAX];
    if (rel_link(prevpath, prev, sizeof prev) != 0)
        die(1, "nothing to roll back");
    char ver[PATH_MAX];
    snprintf(ver, sizeof ver, "%s/%s/VERSION", dir, prev);
    if (access(ver, R_OK) != 0)
        die(1, "the previous version is gone");
    char curlink[PATH_MAX], now[PATH_MAX];
    snprintf(curlink, sizeof curlink, "%s/current", dir);
    int have = rel_link(curlink, now, sizeof now) == 0;
    if (swap_link(dir, "current", prev) != 0)
        die(1, "could not switch current");
    if (have && strcmp(now, prev) != 0)
        swap_link(dir, "previous", now);
    printf("rolled back to %s\n", prev);
    printf("next %s/current\n", dir);
    return 0;
}

static int rate_limited(int http, const char *body)
{
    if (http == 403)
        return 1;
    return body && (strstr(body, "rate limit") || strstr(body, "API rate limit"));
}

/* ---------------------------------------------------------------------------------------------
 * `pxa-update lib ...` -- the licensed PXQN library (libggml-pxqn.so), which updates on its own
 * between engine releases: a signed release manifest from the licence server, per-file sha256s, a
 * staged swap of lib/current that moves both copies of the library at once, and a rollback.
 *
 * The whole decision lives in tools/pxa_lib_update.py. It verifies with the SAME Ed25519 verifier
 * and the same trust anchor as PXA Control's package path, so there is one implementation of the
 * trust decision instead of a second one in C; this tool only finds that script inside the install
 * and hands it the arguments. The install directory is resolved exactly as it is above, so `apply`
 * and `lib apply` can never disagree about which install they are talking about.
 * ------------------------------------------------------------------------------------------- */
#define LIB_USAGE "usage: pxa-update lib check|apply|rollback [--channel stable|beta] [--dir DIR] [--base-url URL] [--key KEY] [--json]"

static const char *install_dir(const char *given, char *buf, size_t buflen)
{
    if (given)
        return given;
    char pkg[PATH_MAX];
    if (own_package(pkg, sizeof pkg) == 0) {
        snprintf(buf, buflen, "%s", pkg);
        char *sl = strrchr(buf, '/');
        if (!sl || sl == buf)
            die(1, "the install has no parent directory");
        *sl = 0;                                /* the install root is the parent of the running package */
        return buf;
    }
    const char *home = getenv("HOME");
    if (!home)
        die(1, "HOME is not set");
    snprintf(buf, buflen, "%s/.local/share/pxa", home);
    return buf;
}

static const char *lib_tool(const char *dir, char *buf, size_t buflen)
{
    const char *env = getenv("PXA_LIB_UPDATE");             /* the tests point this at a scratch copy */
    static const char *rel[] = { "current/tools/pxa_lib_update.py", "tools/pxa_lib_update.py", NULL };
    if (env && env[0]) {
        snprintf(buf, buflen, "%s", env);
        if (access(buf, R_OK) == 0)
            return buf;
    }
    for (int i = 0; rel[i]; i++) {
        snprintf(buf, buflen, "%s/%s", dir, rel[i]);
        if (access(buf, R_OK) == 0)
            return buf;
    }
    char self[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n > 0) {                                            /* a source build keeps the tools beside the binary */
        self[n] = 0;
        char *sl = strrchr(self, '/');
        if (sl) {
            *sl = 0;
            snprintf(buf, buflen, "%s/../tools/pxa_lib_update.py", self);
            if (access(buf, R_OK) == 0)
                return buf;
        }
    }
    return NULL;
}

static int lib_main(int argc, char **argv)
{
    const char *sub = NULL, *dir = NULL, *base = NULL, *channel = NULL, *key = NULL;
    int as_json = 0;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--channel") == 0 && i + 1 < argc)
            channel = argv[++i];
        else if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc)
            dir = argv[++i];
        else if (strcmp(argv[i], "--base-url") == 0 && i + 1 < argc)
            base = argv[++i];
        else if (strcmp(argv[i], "--key") == 0 && i + 1 < argc)
            key = argv[++i];
        else if (strcmp(argv[i], "--json") == 0)
            as_json = 1;
        else if (argv[i][0] != '-' && !sub)
            sub = argv[i];
        else
            die(1, LIB_USAGE);
    }
    if (!sub || (strcmp(sub, "check") != 0 && strcmp(sub, "apply") != 0 && strcmp(sub, "rollback") != 0))
        die(1, LIB_USAGE);
    char dirbuf[PATH_MAX];
    dir = install_dir(dir, dirbuf, sizeof dirbuf);
    char tool[PATH_MAX];
    if (!lib_tool(dir, tool, sizeof tool))
        die(1, "this install has no tools/pxa_lib_update.py; update the engine first");
    char *args[16];
    int n = 0;
    args[n++] = (char *) "python3";
    args[n++] = tool;
    args[n++] = (char *) sub;
    args[n++] = (char *) "--dir";
    args[n++] = (char *) dir;
    if (base) {
        args[n++] = (char *) "--base-url";
        args[n++] = (char *) base;
    }
    if (channel) {
        args[n++] = (char *) "--channel";
        args[n++] = (char *) channel;
    }
    if (key) {
        args[n++] = (char *) "--key";
        args[n++] = (char *) key;
    }
    if (as_json)
        args[n++] = (char *) "--json";
    args[n] = NULL;
    execvp(args[0], args);
    die(1, "could not run python3; the library update needs it");
    return 1;
}

int main(int argc, char **argv)
{
    const char *cmd = NULL;
    const char *base = "https://api.github.com";
    const char *dir = NULL;
    char home_dir[PATH_MAX];
    char pkg[PATH_MAX];
    int from_pkg = 0;
    if (argc > 1 && strcmp(argv[1], "lib") == 0)
        return lib_main(argc - 2, argv + 2);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--base-url") == 0 && i + 1 < argc)
            base = argv[++i];
        else if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc)
            dir = argv[++i];
        else if (strcmp(argv[i], "--rollback") == 0 && !cmd)
            cmd = "rollback";
        else if (argv[i][0] != '-' && !cmd)
            cmd = argv[i];
        else
            die(1, "usage: pxa-update check|apply|rollback [--dir DIR] [--base-url URL]\n" \
              "       pxa-update lib check|apply|rollback [--channel stable|beta] [--dir DIR] [--base-url URL] [--key KEY] [--json]");
    }
    if (!cmd)
        die(1, "usage: pxa-update check|apply|rollback [--dir DIR] [--base-url URL]\n" \
              "       pxa-update lib check|apply|rollback [--channel stable|beta] [--dir DIR] [--base-url URL] [--key KEY] [--json]");
    if (!dir) {
        if (own_package(pkg, sizeof pkg) == 0) {
            from_pkg = 1;
            snprintf(home_dir, sizeof home_dir, "%s", pkg);
            char *sl = strrchr(home_dir, '/');
            if (!sl || sl == home_dir)
                die(1, "the install has no parent directory");
            *sl = 0;
            dir = home_dir;
        } else {
            const char *home = getenv("HOME");
            if (!home)
                die(1, "HOME is not set");
            snprintf(home_dir, sizeof home_dir, "%s/.local/share/pxa", home);
            dir = home_dir;
        }
    }
    if (strcmp(cmd, "rollback") == 0)
        return rollback(dir);
    if (!http_ok(base))
        die(1, "refusing that --base-url");
    char api[1200];
    snprintf(api, sizeof api, "%s/repos/poisonxa16/pxa/releases/latest", base);
    if (mkdir(dir, 0755) != 0 && errno != EEXIST)
        die(1, "could not create the install directory");
    char bodypath[PATH_MAX];
    if (snprintf(bodypath, sizeof bodypath, "%s/.pxa-update-release.json.XXXXXX", dir) >= (int) sizeof bodypath)
        die(1, "install path is too long");
    int fd = mkstemp(bodypath);
    if (fd < 0)
        die(1, "could not store the release list");
    close(fd);
    int http = 0;
    if (fetch(api, bodypath, &http) != 0) {
        char *errbody = slurp(bodypath, NULL);
        unlink(bodypath);
        if (rate_limited(http, errbody)) {
            free(errbody);
            die(1, "GitHub rate limit. Wait a minute and try again.");
        }
        free(errbody);
        die(1, "could not read the PXA release");
    }
    char *json = slurp(bodypath, NULL);
    unlink(bodypath);
    if (!json)
        die(1, "could not read the PXA release");
    if (rate_limited(http, json)) {
        free(json);
        die(1, "GitHub rate limit. Wait a minute and try again.");
    }
    char tag[64];
    if (!json_str_after(json, "\"tag_name\"", tag, sizeof tag)) {
        free(json);
        die(1, "release has no tag");
    }
    char cur[64];
    if (from_pkg) {
        char vp[PATH_MAX];
        snprintf(vp, sizeof vp, "%s/VERSION", pkg);
        version_token(vp, cur, sizeof cur);
    } else {
        read_version(dir, cur, sizeof cur);
    }
    int newer = vercmp(cur, tag) < 0;
    if (strcmp(cmd, "check") == 0) {
        printf("current=%s latest=%s update=%s\n", cur, tag, newer ? "yes" : "no");
        free(json);
        return 0;
    }
    if (strcmp(cmd, "apply") != 0) {
        free(json);
        die(1, "usage: pxa-update check|apply|rollback [--dir DIR] [--base-url URL]\n" \
              "       pxa-update lib check|apply|rollback [--channel stable|beta] [--dir DIR] [--base-url URL] [--key KEY] [--json]");
    }
    if (!newer) {
        free(json);
        printf("current=%s latest=%s update=no\n", cur, tag);
        return 0;
    }
    char url[1024];
    int picked = pick_url(json, url, sizeof url);
    free(json);
    if (picked == -2)
        die(1, "could not read the system C library version; refusing to guess Ubuntu 24.04");
    if (picked != 0)
        die(1, "no engine tarball in that release");
    return apply_release(dir, url);
}
