"""No-hardware PM4 batch-encoding contract test (plan Task 3.1 Step 1).

Compiles a single-translation-unit probe that ``#include "amdev_packets.cpp"``
and builds TWO ``Pm4DispatchConfig`` dispatches — ``timeline_value`` 1 and 2,
``kernargs_va`` 0x6000 and 0x6100 (256 bytes apart), everything else identical —
then concatenates the two 59-dword streams into one 118-dword batch. The probe
prints every dword (8 hex digits each) followed by a ``status: pass`` line after
its own self-check.

The pytest side independently re-asserts the batch-encoding contract:

  * total dwords == 118 (2 x kPm4DispatchDwordCount);
  * RELEASE_MEM type-3 headers at dwords 51 and 110 carry opcode 0x49 in the
    header opcode field (bits [15:8], see pm4_packet3 in amdev_packets.cpp);
  * the RELEASE_MEM timeline-value payload dwords at 56 and 115 are 1 and 2
    (payload order: event, data_sel, lo32(va), hi32(va), value, 0, 0);
  * the two COMPUTE_USER_DATA_0 kernargs VAs differ by exactly 256.

NOTE: the plan text (Task 3.1 Step 1) states the timeline-value slots as
``words[55]``/``words[114]``, but the actual encoder emits them at
``words[56]``/``words[115]`` (RELEASE_MEM payload dword 4 = header + 5). This
test asserts the *actual* encoder behaviour, matching the already-passing
``test_pm4_timeline_contract.py`` (``words[headers[0] + 5]``).
"""

import pathlib
import subprocess
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
INC_DIR = REPO_ROOT / "native_r9700"

# PM4 type-3 header: packet type in bits [31:30], count in [29:16],
# opcode in [15:8] (see pm4_packet3 in amdev_packets.cpp).
K_PACKET3_RELEASE_MEM = 0x49

# One dispatch = kPm4DispatchDwordCount (amdev_packets.cpp:21).
DWORDS_PER_DISPATCH = 59

# RELEASE_MEM is the last packet of each 59-dword dispatch (8 dwords:
# header + 7 payload), so its header sits at 59 - 8 = 51. The second
# dispatch's RELEASE_MEM header is at 51 + 59 = 110.
RELEASE_MEM_HEADER_0 = 51
RELEASE_MEM_HEADER_1 = 110

# RELEASE_MEM payload order (amdev_packets.cpp:178-181):
#   event, data_sel, lo32(timeline_va), hi32(timeline_va), value, 0, 0
# so the timeline value is payload dword 4 = header + 5.
TIMELINE_VALUE_0 = 56
TIMELINE_VALUE_1 = 115

# COMPUTE_USER_DATA_0 SET_SH_REG payload (amdev_packets.cpp:166-168):
#   {0x240, lo32(kernargs_va), hi32(kernargs_va)}
# with the packet header at dword 27, so lo32 is at 29, hi32 at 30.
KERNARGS_LO_0 = 29
KERNARGS_HI_0 = 30
KERNARGS_LO_1 = 88  # 29 + 59
KERNARGS_HI_1 = 89  # 30 + 59

PROBE_SOURCE = r"""
#include <cstdint>
#include <cstdio>
#include <vector>
#include "amdev_packets.cpp"

int main() {
  std::vector<uint32_t> batch;
  for (int stage = 0; stage < 2; ++stage) {
    native_r9700::Pm4DispatchConfig cfg;
    cfg.code_va = 0x5000ULL;
    cfg.kernargs_va = stage == 0 ? 0x6000ULL : 0x6100ULL;
    cfg.timeline_va = 0x7000ULL;
    cfg.rsrc1 = 0x11111111U;
    cfg.rsrc2 = 0x22222222U;
    cfg.rsrc3 = 0x33333333U;
    cfg.workgroup_x = 4;
    cfg.workgroup_y = 5;
    cfg.workgroup_z = 6;
    cfg.global_x = 7;
    cfg.global_y = 8;
    cfg.global_z = 9;
    cfg.timeline_value = stage + 1;
    auto words = native_r9700::build_pm4_dispatch_words(cfg);
    batch.insert(batch.end(), words.begin(), words.end());
  }

  for (uint32_t w : batch) std::printf("%08x\n", w);

  bool ok = (batch.size() == 118);
  ok = ok && (((batch[51] >> 8) & 0xffU) == 0x49U);
  ok = ok && (((batch[110] >> 8) & 0xffU) == 0x49U);
  ok = ok && (batch[56] == 1U) && (batch[115] == 2U);
  const uint64_t va0 = (static_cast<uint64_t>(batch[30]) << 32) | batch[29];
  const uint64_t va1 = (static_cast<uint64_t>(batch[89]) << 32) | batch[88];
  ok = ok && (va1 - va0 == 256ULL);

  if (!ok) {
    std::printf("status: fail\n");
    return 1;
  }
  std::printf("status: pass\n");
  return 0;
}
""".lstrip()


def _compile_and_run(tmp_path: pathlib.Path) -> tuple[str, list[int]]:
    cpp = tmp_path / "pm4_batch_probe.cpp"
    cpp.write_text(PROBE_SOURCE, encoding="utf-8")
    exe = tmp_path / "pm4_batch_probe"
    completed = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(INC_DIR),
            str(cpp),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    run = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    status = lines[-1]
    words = [int(x, 16) for x in lines[:-1]]
    return status, words


def test_pm4_batch_probe_reports_status_pass(tmp_path: pathlib.Path):
    status, _ = _compile_and_run(tmp_path)
    assert status == "status: pass"


def test_pm4_batch_is_118_dwords(tmp_path: pathlib.Path):
    _, words = _compile_and_run(tmp_path)
    assert len(words) == 2 * DWORDS_PER_DISPATCH == 118


def test_pm4_batch_release_mem_headers_carry_opcode_0x49(tmp_path: pathlib.Path):
    _, words = _compile_and_run(tmp_path)
    for idx in (RELEASE_MEM_HEADER_0, RELEASE_MEM_HEADER_1):
        opcode = (words[idx] >> 8) & 0xFF
        assert opcode == K_PACKET3_RELEASE_MEM, (
            f"word[{idx}]=0x{words[idx]:08x} opcode=0x{opcode:02x}, "
            f"expected 0x{K_PACKET3_RELEASE_MEM:02x}"
        )


def test_pm4_batch_release_mem_timeline_values_are_1_and_2(tmp_path: pathlib.Path):
    _, words = _compile_and_run(tmp_path)
    assert words[TIMELINE_VALUE_0] == 1
    assert words[TIMELINE_VALUE_1] == 2


def test_pm4_batch_kernargs_vas_differ_by_256(tmp_path: pathlib.Path):
    _, words = _compile_and_run(tmp_path)
    va0 = (words[KERNARGS_HI_0] << 32) | words[KERNARGS_LO_0]
    va1 = (words[KERNARGS_HI_1] << 32) | words[KERNARGS_LO_1]
    assert va1 - va0 == 256
