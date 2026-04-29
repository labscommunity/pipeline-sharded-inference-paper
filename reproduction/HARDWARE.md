# Hardware setup

## rainier LAN testbed (used for §4, §5, §6.2-§6.7, §6.6 3-stage)

| Alias | Hostname | Internal IP | Role | Hardware |
|---|---|---|---|---|
| node-02 / `alpha` | alpha.lan | 192.168.86.250 (Wi-Fi disabled, on Ethernet) | coord (stage_0) | HP OmniBook X 16, Core Ultra X7 358H (Panther Lake), Arc B390 iGPU 17 GB, 32 GB DDR5 |
| node-01 / `charlie` | charlie.lan | 192.168.86.28 | worker (stage_1) | ASUS Zenbook S 14, Core Ultra 7 258V (Lunar Lake), Arc 140V iGPU 16 GB, 32 GB LPDDR5X-8533 |
| node-00 / `beta` | beta.lan | 192.168.86.36 | worker (stage_2) | ASUS Zenbook S 14, Lunar Lake / Arc 140V, 32 GB |

SSH key: `~/.ssh/cascadia_ed25519`, user `cascadia`, all on Ethernet (Wi-Fi blocks the 3-stage TCP retransmit path with `WinError 10053` — see §8.4).

## Tiber Cloud fleet (used for §6.8 Tiber DERP, §6.10 70B 4-stage)

Tiber instances are reachable only through per-account bastions; SSH config blocks below. See `cascadia-fleet/docs/INTEL_TIBER.md` for the full bring-up runbook.

```
# ~/.ssh/config

# Bastions (Intel-side jump hosts)
Host cascadia-jump-217
    HostName 192.55.48.217
    User guest
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /dev/null

Host cascadia-jump-218
    HostName 192.55.48.218
    User guest
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /dev/null

# AI PC instances (used by paper)
Host cascadia-matias-01      # 70B 4-stage LL coord; §6.8 8B 2-node coord
    HostName 192.168.15.2
    User devcloud
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump cascadia-jump-218
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /dev/null

Host cascadia-matias-02      # 70B stage_1 worker; §6.8 8B 2-node worker
    HostName 192.168.18.2
    User devcloud
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump cascadia-jump-218
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /dev/null

Host cascadia-pawan-01       # 70B stage_2 worker
    HostName 192.168.19.2
    User devcloud
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump cascadia-jump-218
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /dev/null

Host cascadia-pawan-02       # 70B stage_3 worker
    HostName 192.168.20.2
    User devcloud
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump cascadia-jump-218
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /dev/null

Host cascadia-tate-04        # 70B PL coord variant (Battlemage Xe3 iGPU)
    HostName 192.168.11.2
    User devcloud
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump cascadia-jump-218
    StrictHostKeyChecking accept-new
    UserKnownHostsFile /dev/null
```

## Tailscale tailnet

The Tiber instances communicate through Tailscale's DERP relay (Seattle region, ~16 ms relay-mediated RTT) because direct UDP/41641 between AI PC instances is blocked by Intel's network. Each AI PC must be authenticated to the same tailnet with `tag:fleet`. See `cascadia-fleet/docs/INTEL_TIBER.md` for the full setup.

After Tailscale auth, the Tailscale IPs (visible via `tailscale status`) are used for cross-instance traffic in the coord scripts:

| Tiber alias | Tailscale IP (example, varies per tailnet) |
|---|---|
| matias-01 | 100.88.94.47 |
| matias-02 | 100.77.178.45 |
| pawan-01 | (assigned at join) |
| pawan-02 | (assigned at join) |
| tate-04 | (assigned at join) |

The reproduction scripts read these IPs from `configs/tiber_ips.env`.

## Software

* OpenVINO 2026.1.0+ (2026.2.0 also tested for matias-02; not 2026.0)
* Python 3.11.9 (rainier nodes), 3.14 (Tiber Lunar Lake instances) — both work
* `transformers` 4.57.6 / 5.5.4 / 5.6.2 (varies; all produce compatible exports)
* `torch` 2.11.0+xpu (rainier alpha) / 2.11.0+cpu (rainier beta)
* `nncf` for INT4 weight compression
* `tailscale` (Tiber-only)

`scripts/run/setup_env.sh` checks all of these and reports any missing.
