# Anibal store: licd 403 hard-lock (shared spec)

Rulings (PXANetwork, 2026-10-19 04:01 and 04:33): Control's strict rule is the spec; the lock time is
on licd's clock. RC3.6 gate items (04:48): a corrupt cache file fails closed (F305-P), one cross-process
lock around every load-modify-save (F303-P), and the not-yet check on licd's clock. RC3.6 gates
(05:06): the lock journal outside grants.json (F309-P), the bounded lock time (F310-P) and a
first-time install.key created once under a lock. Core's gate (anibal/a-post35 0cabc87f61) and the Plugins host (plugins/v1-store-rc36
11c2358bbd) adopt it in RC3.6. Reference: `tools/pxa_control/store/__init__.py` (`server_time`,
`EntitlementCache.hard_lock`, `.add`, `._load`, `._corrupt`, `file_lock`, `write_atomic`, `LicdClient.sync`)
on the store-verify branch.

## Trigger
- licd answers **403** with JSON `code` in `revoked | not_linked | account_disabled`, on
  `/v1/store/refresh` **or** on the `/v1/store/nonce` request before it.
- Record the lock time `hard_lock_at` on licd's clock: the 403 JSON body's `server_time` (epoch
  seconds or ISO 8601 UTC) if present and readable, else the HTTP `Date` header. **F310:** when both
  parse and differ by more than 5 minutes, the EARLIER of the two (`server_time_checked`). Only when
  neither parses (or is outside 2017..2099, inf/nan/overflow included): the local clock minus 5 minutes.
  Record `hard_lock_clock`: `server` | `local`, and `hard_lock_source`: `body` | `date` | `local`.
- **F310 cap:** a lock time is never later than the local clock + 24 h; a capped one records
  `hard_lock_capped: true`. The cap holds until a fresh grant lifts the lock (the record then goes).
  A second 403 moves `hard_lock_at` later (within the cap), never earlier.
- Record every grant id seen so far (`hard_lock_seen`: held, newest-issued and revoked ids).

## Effect
- Lock every item: everything held, already locked, or listed in the last account answer.
- Drop every grant. Keep the revocation list and the anti-replay records (`newest`).
- Write each lock to the shared `locks` key (what Core and Plugins read) and to Control's
  `hard_locks` key. A later good refresh may rewrite `locks` from owned[], but not `hard_locks`.

## Unlock (only a freshly issued valid grant)
While a hard-lock is in force, a grant for ANY item counts only if all of these hold:
1. it verifies (signature, kid, times, this machine);
2. it is not revoked;
3. its `issued` (licd-issued, so the same clock as a server lock time) is after `hard_lock_at`;
4. **(v4, replaces the seen-id and nonce rules; see "Hard-lock v4" below)** EITHER its `ent_id` was
   known at lock time and its `issued` is strictly newer than the newest `issued` recorded for that
   `ent_id` at lock time (`hard_lock_newest`), OR its `ent_id` is new (licd issues a new one only
   after a relink).
   ~~its grant id (`ent_id`) was not seen before the lock, nor (when the grant carries one) its
   `nonce` (`hard_lock_seen` holds `nonce:<n>` entries; Control applies this since RC3.6, as Plugins).~~
   *(superseded by v4, PXANetwork 05:18)*
Otherwise it is refused (`pre_lock`; an older copy of a known id stays `replayed`).
Re-sending the old grant never unlocks (same `issued`), and neither does a refresh that marks items
active without a grant. A re-enabled account's refresh, which licd re-signs on the SAME `ent_id`
with a newer `issued`, does unlock (v4). A good grant lifts its own item only; the lock record
(`hard_lock_at`, `_seen`, `_clock`, `_newest`) goes once no item is hard-locked. With the local
fallback the newest-issued rule still blocks every replay.

## Hard-lock v4 (PXANetwork 05:18, from licd store.py, hive #27111; Plugins additions; supersedes v3)
licd re-signs the same `ent_id` on every refresh with a new `issued`, so "a seen id never unlocks"
kept a re-enabled account locked for good, and licd grants carry no nonce. v4:
- While hard-locked a grant counts as fresh when it verifies, is not revoked, matches this machine's
  `install_pub` (else `wrong_machine`), its `issued` is after `hard_lock_at` (licd's clock), and
  (a) its `ent_id` was known at lock time and its `issued` > `hard_lock_newest[ent_id]`, or
  (b) its `ent_id` was not known at lock time (relink).
- Anti-replay = newest `issued` per `ent_id` (`newest`) + licd's server time. No nonce is used; the
  v3 nonce entries are no longer written or read (Control). The rule "seen ent_id always refused" is dropped.
- Lock time: `hard_lock_newest` = per-`ent_id` max of `newest` and every held grant's `issued`;
  `hard_lock_seen` stays the set of ids known at lock time.
- Journal (outside grants.json, both copies): adds `newest` {ent_id: issued} (max-merged, the 20000
  newest kept), `hard_lock_newest` {ent_id: issued} while locked, and `licd_high_water` (latest licd
  time seen: reply server_time / Date, 403 lock time; never local; capped at now + 24 h). Corrupting
  grants.json cannot erase them. **Fail closed:** a known `ent_id` with no `hard_lock_newest` record
  (a pre-v4 lock, or the record stripped) is never fresh; only a new `ent_id` can then unlock.
- Corruption lock (`corrupt_clock_fast` fix): `hard_lock_at` = min(max(journal `hard_lock_at`,
  `licd_high_water`), now + 24 h) with `hard_lock_clock: server`; the local clock only when no licd time
  is on record (`local`). A computer whose clock runs a month ahead is unlocked by licd's next fresh
  grant; a clock set back (F309) still cannot bring an old grant back (its `issued` is not newer).
- Key names (final, RC3.6 item 4; Control, Plugins and Core): `licd_high_water`, `newest`,
  `hard_lock_newest`, `revoked` (Control constants `J_LICD_SEEN`, `J_NEWEST`, `J_LOCK_NEWEST`, `J_REVOKED`).
- Revoked ids are journalled (F318, grow-only), merged before any sync result and kept by every rebuild.
- ~~v3 (interim, nonce): a seen `ent_id` with a fresh nonce unlocks~~ *(superseded by v4; Core's v3
  section, if added, is kept below as written)*.

## Not-yet check on licd's clock (RC3.6)
- A grant whose `issued` is more than 5 minutes (CLOCK_SKEW_S = 300) after **licd's clock** is
  refused `not_yet`. licd's clock = the `server_time` in the JSON body of the reply that carried the
  grant (refresh, link, voucher, redeem), else that reply's HTTP `Date` header (same parser,
  2017..2099 range and F310 earlier-of-the-two rule as the lock time). Only when neither parses: the
  local clock.
- Expiry (`not_after`), `refresh_due` and the 48 h rollback guard stay on the local clock.
- A grant accepted on licd's clock keeps counting when licd is ahead of this computer: the cache
  records licd's time per grant in `licd_time` ({ent_id: t}) and later reads check not-yet against
  max(local clock, that time). A component that re-verifies grants on read must do the same (or skip
  not-yet for stored grants), else a grant from a licd running ahead is dropped on the next read.

## Shared file: one lock, atomic replace (F303-P)
Contract: /workspace/plugins-notes/F305-control-corrupt-fail-closed.md (Plugins `_StoreLock` /
`_os_lock`, plugins/v1-store-rc36 9351f0a99c). Control follows it as below.
- `<config>/store/grants.json` is shared by Control, the plugin host and Core's gate. Every
  load-modify-save (add, hard_lock, set_locks, drop_item, apply_revocations) loads INSIDE one
  exclusive lock on `<config>/store/grants.json.lock`, opened `O_RDWR|O_CREAT|O_NOFOLLOW`, 0600,
  never deleted: POSIX `fcntl.flock(LOCK_EX)` on the whole file (not `lockf`/`F_SETLK`), Windows
  `msvcrt.locking(LK_NBLCK)` on byte 0 after `lseek(0)`, retried. Re-entrant within one thread (not
  across fork). Control waits up to 30 s on both.
- A busy or unopenable lock (timeout, a symlink planted as the lock file) is never an unlock: Control
  refuses `add` / `set_locks` (`StoreBusy` / OSError, nothing changes); a call that only adds locks or
  only reads (`hard_lock`, `drop_item`, `apply_revocations`, reads) still runs under the in-process
  lock, so a 403 is always written.
- Reads never write (`has`, `active`, `items`, `locks`, `status`, the view): `seen` (rollback guard)
  moves on every write (each sync writes), never on a read. The one exception is the fail-closed corruption write, under the lock.
- Read the clock (and make any other call that may reach another component) BEFORE taking the lock.
- Keep keys you don't know: Control saves the dict it loaded, so the plugin host's `fresh`,
  `star_held`, `hard_lock_why`, `hard_lock_clock`, `last_sync`, `revoked_items` survive. Since RC3.6
  (F309) Control bumps `gen` and writes the lock journal on every save, exactly as the plugin host
  (contract option 2, merge-and-grow; next section).
- Save = write to a temp file in the same directory, flush, fsync, chmod 0600 (`os.chmod`, not
  `os.fchmod`: Windows Python 3.12), `os.replace` onto grants.json, fsync the directory. A crash
  leaves the old file or the new one, never part of one; a failed save removes its temp file.
- `account.json` and `catalog.json` (Control only) are saved the same way under their own `.lock`.

## Corrupt file: fail closed (F305-P, = Plugins' GrantCache rule)
- Missing grants.json: empty state (no lock), unless a lock journal exists
  (`<config>/store/grants.journal.json` or `<config>/plugins/store-lock-journal.json`): then fail closed
  as below (`corrupt_why "missing"`, nothing to keep aside). A symlink or a non-regular file at
  grants.json is corrupt (`"link"`): lstat first, open `O_NOFOLLOW`; the link itself is moved aside,
  its target is never read or written. Unreadable, not UTF-8 JSON, cut off, not an object, or
  a known key of the wrong type (`grants` dict, `revoked` list, `newest` dict, `seen` number, `locks`
  dict, `hard_locks` dict, `last_sync` number, `revoked_items` dict, `licd_time` dict, `fresh`
  list; `hard_lock_at` a number with a `hard_lock_seen` list) is CORRUPT.
- Corrupt: under the lock, keep the file aside as `grants.json.corrupt-<unix time>` (a `-<n>` suffix
  if that name exists; never over an earlier copy; moved, else copied), then write a new grants.json:
  `hard_locks {"*": "corrupt"}`, `hard_lock "corrupt"`, `hard_lock_at` = local now,
  `hard_lock_clock "local"`, `hard_lock_seen []`, `fresh []`, `corrupt_copy` = the copy's name.
  If the file cannot be kept aside, nothing is written (the bad file stays byte for byte) and every
  item reads locked.
- `"*"` locks every item not in `fresh`; a fresh grant issued after `hard_lock_at` lifts its own item
  (added to `fresh`). Control never pops `"*"` itself (the plugin host clears it per its F306 rule or
  on a good signed account answer). The anti-replay records went with the bad file.
- A fresh corruption file carries no `gen`, so the plugin host also reads it as older than its journal
  and fails closed with `"*"` (safe: each item then needs a fresh grant or a good sync).

## The lock journal outside grants.json (F304 / F309-P, = Plugins GrantCache, DESIGN.md s31.3)
- Two copies, same content: `<config>/store/grants.journal.json` and `<config>/plugins/store-lock-journal.json`
  (next to install.key). Format `{v: 1, gen, hard_locks {item: code} (no "*"), star (bool), fresh [],
  held [items ever held], ids [every grant id, and "nonce:<n>", ever held/newest/revoked],
  seen (high-water time), hard_lock_at?, hard_lock_seen?, hard_lock_clock?}`. `seen` is Control's
  addition (Plugins must keep it; see below).
- Every save, under the store lock: re-read the journal and merge it into the state, `gen` =
  max(file gen, journal gen) + 1, write grants.json (temp + fsync + os.replace + dir fsync), then
  both journals the same way (one copy is enough if the other is on a read-only share).
- The journal only grows: locks are unioned, an item leaves it only when a fresh verified grant for it
  (issued after the lock, id/nonce never seen) is in grants.json; `ids`, `held`, `hard_lock_seen` are
  unioned; `hard_lock_at` and `seen` take the max. It is a whole-file atomic replace, not a log of
  appended lines (that is the Plugins format the contract pins); "append-only" holds as grow-only.
- Every load reads the journal first. Fail closed (`"*"`, as a corrupt file, `cache.why`):
  grants.json corrupt / symlink (`link`), missing while a journal exists (`missing`), `gen` older than
  the journal (`rolled_back`: a restored copy or a writer that skipped the lock), or a journal that
  is there but unreadable (`journal`). A missing file with no journal is a first start.
- **F309 rebuild:** a corruption lock is rebuilt from the newest journal: its per-item 403 locks
  (merged back as single-item locks once `"*"` is lifted for an item), every journalled id as
  `hard_lock_seen`, `newest` and `hard_lock_newest` (v4), `gen` from the journal, and `hard_lock_at`
  on licd's clock (v4: min(max(journal `hard_lock_at`, `licd_high_water`), now + 24 h); local now only
  with no licd time on record). So setting the clock back into the 5-minute skew of an old grant and
  then corrupting grants.json cannot bring that grant back: its `issued` is not newer than the
  journalled newest for its id.
- **High-water time:** the journal's `seen` is merged into the state on every load, so the 48 h
  rollback guard (`clock_rollback`, every grant expired) survives a corrupted or deleted grants.json.
- What it cannot do: an owner who deletes or edits grants.json and both journals lifts the lock
  offline; the next licd contact answers 403 again and re-locks.

## install.key: created once (RC3.6)
- `<config>/plugins/install.key` (32 bytes, 0600). A valid key is read without a lock. The first-time
  create runs under `<config>/plugins/install.key.lock` (same `file_lock` primitive), re-reads, and
  publishes a fully written, fsynced temp file by **hard link** (fails if the key exists: the existing
  key wins and is returned; no reader ever sees a partial key). No hard links: `O_CREAT|O_EXCL`.
  Only a key file of the wrong length is replaced (under the lock). Two processes starting at once
  end up with the same key.

## Answers that change nothing (no lock, no unlock)
401, 404, 429, 5xx, a 403 with any other code (e.g. `requires_item`), a 403 whose body is not
licd's JSON (e.g. a Cloudflare page), offline/garbled/cut-off replies, and malformed replies.

## Entitlements view
Keys: `locks` ({item_id: reason}, every lock in force; `"*"` = all), `hard_lock` (the 403 code, or
`"corrupt"` for a corruption lock, or null; a 403 code is cleared by the next good refresh) and,
since RC3.6, `cache`: `{state: ok | hard_locked | corrupt, hard_lock, corrupt_copy, since, clock,
why: corrupt | link | missing | rolled_back | journal | null}`.
Consumers that check the exact key set must add all three.

## Test cases (tests/test-pxa-control-store-verify.py; HardLock403 unless named otherwise)
- each_hard_lock_code_locks_every_item_and_drops_every_grant: each code locks all 3 items, grants {}, both keys written, view empty, `newest` kept
- a_403_on_the_nonce_step_locks_too: 403 account_disabled on nonce locks everything
- the_lock_time_and_the_grant_ids_seen_are_recorded: `hard_lock_at` = licd's Date, `server`, held ids seen
- a_second_403_moves_the_lock_time_later: later Date grows `hard_lock_at`; an earlier one leaves it
- server_date_ahead_of_the_local_clock: Date now+200: lock = that, `server`; issued now+150 no, now+260 unlocks
- server_date_behind_the_local_clock: Date now-3600: issued now-4000 no, now-3000 unlocks
- no_date_falls_back_to_local_minus_5_minutes: `local`, lock = now-300; now-400 no, now-200 unlocks
- unparseable_date_falls_back_the_same_way: garbage, impossible, 1970 Dates: as above
- a_server_time_in_the_403_body_wins_over_the_date_header: epoch and ISO body times win; unreadable one -> Date
- locked_then_offline_stays_locked_with_no_state_change: 503 store_offline, cache file identical
- locked_then_malformed_or_garbled_stays_locked: garbled, cut-off, non-JSON, owned[] without status, grants not a list: file unchanged
- other_refusals_and_trouble_never_lock: 403 requires_item/bad_signature/empty code, 401, 404, 429, 500, 403 HTML: raises, nothing changes
- Transport.test_offline_and_malformed_syncs_change_nothing: garbled, cut-off, non-JSON, non-object: 503, nothing changes
- resending_the_old_grant_never_unlocks: direct add and in a refresh reply: `pre_lock`, still locked
- an_older_copy_of_the_old_grant_is_a_replay: `replayed`, still locked
- ~~the_old_grant_id_with_a_new_issued_at_never_unlocks~~ (superseded by v4) → the_old_grant_id_with_the_same_issued_never_unlocks: `pre_lock`
- a_new_grant_id_issued_before_the_lock_never_unlocks: `pre_lock`, still locked
- a_refresh_marking_items_active_without_a_grant_never_unlocks: no grants, all locks stay
- a_grant_that_does_not_verify_never_unlocks: forged statement refused, still locked
- a_revoked_fresh_grant_never_unlocks: fresh grant revoked in the same reply: still locked
- a_fresh_grant_unlocks_only_its_item: new id, issued after: that item only; record stays for the rest
- the_lock_record_goes_once_every_item_is_unlocked: `hard_locks`, `locks`, `hard_lock_at`, `hard_lock_seen` all cleared
- while_locked_an_old_grant_for_an_unlisted_item_is_refused_too: an id seen before the lock: `pre_lock`
- the_auto_refresh_in_the_entitlements_route_applies_the_lock: sync() on 403 not_linked returns 0 grants, empty view
- test-pxa-control-store Routes.test_entitlements_shape: view keys include `locks`, `hard_lock`, `cache` (state ok)
- OwnedLocks.old_cache_files_without_locks_still_load: no `locks` key loads; `locks` of the wrong type -> corrupt
- CorruptFailsClosed.corrupt_json_fails_closed_and_is_kept_aside: `{garbage` -> `*` lock, bytes kept in the copy, old grant `pre_lock`, view `corrupt`
- CorruptFailsClosed.truncated_file_fails_closed: half a file -> the same
- CorruptFailsClosed.non_object_and_wrong_types_fail_closed: 18 shapes ([] / "x" / 42 / null / empty / bad UTF-8 / wrong-typed keys / deep nesting)
- CorruptFailsClosed.every_entry_point_fails_closed: add, set_locks, items, drop_item, hard_lock, active, apply_revocations, status
- CorruptFailsClosed.a_file_that_cannot_be_kept_aside_is_never_overwritten: bad bytes unchanged after every call, still locked
- CorruptFailsClosed.a_fresh_grant_lifts_the_corruption_lock_for_its_item_only / two_corruptions_never_overwrite_an_earlier_copy
- CorruptFailsClosed.a_plugin_host_corruption_lock_is_honoured / a_missing_file_is_empty_not_corrupt
- CorruptFailsClosed.a_symlinked_file_fails_closed_and_its_target_is_untouched: link moved aside, `why "link"`
- CorruptFailsClosed.a_missing_file_with_a_lock_journal_fails_closed: either journal path, `why "missing"`
- OneWriterAtATime.concurrent_writers_in_processes_lose_no_update: 24 processes add 24 grants, all kept
- OneWriterAtATime.concurrent_writers_in_threads_lose_no_update: 16 threads, 4 caches, adds + set_locks, nothing lost
- OneWriterAtATime.add_racing_a_403_in_another_process_keeps_the_lock: add() paused inside the lock; the 403 process waits, then locks on top
- OneWriterAtATime.a_403_landing_inside_add_like_qa_repro_keeps_the_lock: QA F303-P shape (403 from add()'s clock call)
- OneWriterAtATime.add_waits_for_a_403_in_progress_and_is_refused: hard_lock paused mid-save; add() waits, then `pre_lock`
- OneWriterAtATime.the_lock_file_is_shared_and_kept: another process waits on grants.json.lock
- OneWriterAtATime.a_busy_lock_refuses_an_unlock_but_a_403_is_still_written: add/set_locks `StoreBusy`, file unchanged; hard_lock lands
- OneWriterAtATime.a_symlinked_lock_file_is_never_followed: add refused, hard_lock lands, link target untouched
- OneWriterAtATime.reads_never_write: active/has/items/locks/status/view leave the file (same inode)
- OneWriterAtATime.the_plugin_hosts_keys_are_kept: gen, star_held, hard_lock_why, last_sync, revoked_items, unknown keys survive
- CrashMidWrite.a_failure_before_the_rename_leaves_the_old_file_and_no_temp: fsync / replace fail: old file, no temp
- CrashMidWrite.a_process_killed_mid_write_leaves_the_old_file_intact: os._exit before the rename: old file whole, lock free
- CrashMidWrite.saves_are_0600_and_go_through_one_rename
- LicdClockNotYet.licd_20_minutes_ahead_accepts_a_fresh_grant: Date (or body server_time) now+20 min, issued now+20 min: accepted, still counts
- LicdClockNotYet.a_grant_more_than_5_minutes_past_licd_time_is_refused: server+400 s, licd 20 min behind, local ahead: `not_yet`
- LicdClockNotYet.without_licd_time_the_local_clock_applies: no / unparseable Date: issued now+20 min `not_yet`, now+200 s accepted
- LicdClockNotYet.the_reply_carries_licd_time_and_add_takes_it: `_Reply.server_time`, `add(server_now=)`, `licd_time` recorded
- JournalF309.qa_repro_clock_back_into_the_skew_then_corrupt: QA F309 probe: 403, clock to old issued-299, `{garbage`: old grants `pre_lock`, locked
- JournalF309.without_a_403_an_old_held_grant_never_lifts_a_corruption_lock: journal `ids` count as seen
- JournalF309.the_journal_holds_gen_ids_locks_and_the_high_water_time: both copies equal, gen = file gen, ids/held/seen/hard_locks
- JournalF309.a_corrupt_or_missing_file_is_rebuilt_from_the_journal: corrupt / missing / truncated: `*`, lock time kept, ids seen, 403 lock stays journalled
- JournalF309.a_restored_older_copy_is_rolled_back_and_fails_closed: older gen -> `rolled_back`, old grant `pre_lock`
- JournalF309.the_high_water_time_survives_a_corruption: clock 3 days behind the journal's `seen` after corruption: `clock_rollback`
- JournalF309.a_garbled_journal_fails_closed_and_one_journal_is_enough / a_plugin_host_journal_lock_is_merged_in
- LockTimeF310.server_time_checked: agree -> body; disagree > 5 min -> earlier; one missing -> the other
- LockTimeF310.a_body_years_ahead_of_a_right_date_locks_at_the_date: QA future_server_time shape; a fresh grant unlocks
- LockTimeF310.a_date_ahead_of_the_body_by_more_than_5_minutes_locks_at_the_body / agreeing_body_and_date_keep_the_body
- LockTimeF310.a_lock_time_is_capped_at_now_plus_24_hours_until_a_fresh_grant: Date +30 d -> now + 24 h, capped; licd's fresh grant lifts it
- LockTimeF310.direct_hard_lock_is_capped_too / qa_insane_times_are_none (QA sane_time list: inf, nan, 10**400, +99:00, year 1, 99999 ...)
- InstallKeyOnce.processes_starting_at_once_get_the_same_key: 5 rounds x 16 forked processes: one key, 32 bytes, 0600, no temp left
- InstallKeyOnce.threads_starting_at_once_get_the_same_key / a_key_published_by_another_writer_wins_and_is_never_overwritten / a_wrong_length_key_is_replaced_once

### Hard-lock v4 tests (HardLockV4, a HardLock403 subclass so every HardLock403 case also runs under it)
- a_reenabled_account_same_ent_id_newer_issued_unlocks: same ent_id re-signed after the lock unlocks its item only
- same_ent_id_with_an_equal_or_older_issued_is_refused: equal `pre_lock`, older `replayed`
- a_newer_issue_before_the_lock_time_is_refused: newer than the record but not after hard_lock_at: `pre_lock`
- a_new_ent_id_after_relink_unlocks
- a_revoked_grant_never_unlocks: a revoked id re-signed newer: `revoked`; a new id revoked in the same reply stays locked
- a_replay_of_the_old_grant_is_refused: by add and in licd's reply
- a_newer_issue_for_another_machine_is_refused: `wrong_machine`
- the_lock_time_newest_survives_a_corrupt_file: journal holds newest / hard_lock_newest / licd_high_water
- a_missing_lock_time_newest_record_fails_closed_for_known_ids: known id refused, new id unlocks
- corrupt_clock_fast_locks_on_licd_time: clock a month ahead + corrupt file: `server` lock at licd time, fresh grants accepted
- without_licd_time_on_record_a_corruption_locks_at_the_local_clock
