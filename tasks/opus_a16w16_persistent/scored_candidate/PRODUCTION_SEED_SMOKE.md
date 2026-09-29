# OPUS production-header seed smoke

**One public bucket only; unscored.** On 2026-09-29, the unchanged pinned
persistent header was installed into a clean AITER overlay and compared with
pinned AITER through two isolated production `opus_bmm` workers on one
MI350X/gfx950. Each worker had a new, disjoint `AITER_JIT_DIR` and a separate
Python import cache. The public `aligned-nooob` `(8192,4096,256)`, kid 1300
case passed six reuse/guard/oracle steps per worker, exact persistent
profiler dispatch, a random BF16 graph-input hash match, direct and graph
prechecks, and **read-only** post-graph oracle/guard/input checks. The
32-call graph replay used 20 serialized alternating pairs and only the two
known worker PIDs; no foreign GPU PID was seen at its gates.

| AITER us/call | Overlay us/call | Overlay/AITER | Relative MAD AITER/overlay | Raw SHA256 |
| ---: | ---: | ---: | ---: | --- |
| 74.752439 | 73.514920 | .983445 | .016329 / .010398 | `f2071bbedfad65944125bc7114d6b8e5b4e7ae362e183dbcf082abbfb4c13f8d` |

The two JIT module SHA256 values were
`f923ae1240d55b60d82a74e2ebb23f178de7991c77475d61635d34ccb1b4a69a`
and `c88050693ceb686b94cfbce337869f05223d7f415ed55e229feb2484ea7850a9`.
Both module origins were verified inside their own cache, but the binary
hashes differ despite unchanged header content. This may reflect build-path
or other nondeterministic content; it has not been investigated and is **not**
evidence of different mathematical kernel behavior. The candidate overlay
had no changed tracked paths. An unchanged-header single-bucket ratio below
one is measurement variation, not an optimization gain.

The first owned-container attempt stopped before any kernel because the
host's LDAP UID was absent from the container's mounted `/etc/passwd`.
Its raw SHA256 is
`1e1f2ad317e3ec9563e9b2c9431d407a56d3e9427aea875fc333d763d654aece`;
status `inconclusive_execution_error`. A separate no-GPU container check
validated the corrected private minimal NSS mapping (`getpwuid` resolves to
`aiter-replay`) before the passing retry. This failure is infrastructure,
not candidate correctness evidence. Both attempt roots and logs remain
private off-repo.

Provenance: local task branch commit `31d20ce` (on `0f2795c` and
`c9ad7e5`); pinned AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`;
unchanged header SHA256
`bf621cfe97d2b38a89ca668c57b32a10094aeef92b4fdbb723e530e0abd6946b`;
task JSON SHA256 `e94d748a44228639fc0e056ed5077e9a789fc6d2ebc03237124de9cf35b5b13c`;
public matrix SHA256 `840326059a47f5ec7f61006101f06b3061aa0ce2d1022e7805c75a2389adc77c`;
production driver SHA256
`bc8c41e223cc53cc856198efa7460f5716b009e98001c9439c795f9c6985cc91`;
compiler image ID
`sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`.
The exact eight-public plus twelve-withheld production matrix and a second
fresh production session are still unrun. Pre/post PID checks cannot rule
out short-lived overlap between pairs. `scored_eligible=false`; no agent
candidate or agent error incidence was assessed.
