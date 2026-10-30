"""The compiled pxa-control binary's entry point (packaging/control/build-linux.sh, BUILD-CONTROL.md)."""
import os
import sys

# No __pycache__ next to Control's sources when this entry runs from a source tree (review N3); the frozen build
# imports from its archive and writes none either way. Python children inherit the switch.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def _ca_bundle():
    """The compiled build carries the build image's OpenSSL, whose built-in CA path (/usr/lib/ssl on Ubuntu) is missing
    on other distributions, so every https call (PXA Store, Hugging Face, library updates) would fail certificate
    checks there. When that default finds nothing and the user set nothing, point SSL_CERT_FILE at the system bundle.
    macOS: the python.org build looks in /Library/Frameworks/Python.framework/... (absent on a user's Mac, and empty
    on the build Mac until "Install Certificates" is run), so https failed there too; /etc/ssl/cert.pem is on every Mac."""
    if sys.platform not in ("linux", "darwin") or os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return
    import ssl
    p = ssl.get_default_verify_paths()
    if (p.cafile and os.path.isfile(p.cafile)) or (p.capath and os.path.isdir(p.capath) and os.listdir(p.capath)):
        return
    for f in ("/etc/ssl/certs/ca-certificates.crt",          # Debian, Ubuntu, Arch, Gentoo, Unraid
              "/etc/pki/tls/certs/ca-bundle.crt",            # Fedora, RHEL, CentOS
              "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
              "/etc/ssl/ca-bundle.pem",                      # openSUSE
              "/etc/ssl/cert.pem"):                          # Alpine, Void
        if os.path.isfile(f):
            os.environ["SSL_CERT_FILE"] = f
            return


if __name__ == "__main__":
    _ca_bundle()
    from pxa_control.desktop import main
    sys.exit(main())
