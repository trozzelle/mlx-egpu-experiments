"""No-hardware contract tests for the native AMDev/SDMA transfer probe."""

from pathlib import Path
import subprocess


PROBE_SOURCE = Path("experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp")

REQUIRED_LOG_FIELDS = (
    "runtime_substrate",
    "pci_id",
    "arch",
    "transfer_byte_count",
    "cpu_comparison_status",
    "host_device_transfer_status",
    "failure_stage",
    "failure_text",
    "exit_status",
)

EXPECTED_REMOTE_CMD_FRAME_HEX = (
    "0251750000050000000807060504030201887766554433221100ffeeddccbbaa99"
)

EXPECTED_SYSMEM_PAGE_LIST_LINES = (
    "self_test: sysmem-page-list",
    "requested_page_count: 4",
    "raw_pair_count: 3",
    "terminator_pair_index: 3",
    "expanded_page_count_before_truncation: 6",
    "parsed_page_count: 4",
    "truncated: yes",
    "page_0_paddr: 0x0000000100000000",
    "page_1_paddr: 0x0000000100001000",
    "page_2_paddr: 0x0000000200000000",
    "page_3_paddr: 0x0000000200001000",
    "status: pass",
)

EXPECTED_SDMA_PACKET_ENCODING_LINES = (
    "self_test: sdma-packet-encoding",
    "packet_dword_count: 7",
    "transfer_byte_count: 32",
    "copy_count_field: 31",
    "source_address: 0x0102030405060708",
    "source_address_le: 0807060504030201",
    "destination_address: 0x1122334455667788",
    "destination_address_le: 8877665544332211",
    "packet_hex: 010000001f0000000000000008070605040302018877665544332211",
    "status: pass",
)

EXPECTED_AM_VM_PTE_ENCODING_LINES = (
    "self_test: am-vm-pte-encoding",
    "leaf_level: PTB",
    "gfx_ip_major: 12",
    "mtype_uc: 3",
    "sysmem_leaf_flags: 0x80c0000000000077",
    "vram_leaf_flags: 0x8000000000000071",
    "table_entry_flags: 0x0000000000000001",
    "sysmem_staging_pte: 0x80c0000080000077",
    "vram_pte: 0x8000000006000071",
    "sysmem_readback_pte: 0x80c0000080008077",
    "sysmem_sdma_control_pte: 0x80c0000080010077",
    "status: pass",
)

EXPECTED_AM_VM_PAGE_TABLE_PLAN_LINES = (

    "self_test: am-vm-page-table-plan",
    "va_base: 0x0000200000000000",
    "staging_va: 0x000020003fe00000",
    "staging_byte_count: 1048576",
    "staging_ptb_page_count: 256",
    "vram_va: 0x0000200000001000",
    "readback_va: 0x0000200000002000",
    "va_shifts: 12,21,30,39",
    "first_level: PDB2",
    "pdb2_index: 0",
    "pdb1_index: 0",
    "pdb0_index: 511",
    "staging_ptb_index: 0",
    "vram_ptb_index: 1",
    "readback_ptb_index: 2",
    "boot_arena_size: 0x02000000",
    "ptable_arena_base: 0x02000000",
    "fixed_vram_buffer_paddr: 0x0000000006000000",
    "sdma_control_va: 0x0000200000003000",
    "sdma_control_ptb_index: 3",
    "kernel_output_va: 0x0000200000004000",
    "kernel_output_ptb_index: 4",
    "kernel_code_va: 0x0000200000005000",
    "kernel_code_ptb_index: 5",
    "kernel_kernargs_va: 0x0000200000006000",
    "kernel_kernargs_ptb_index: 6",
    "compute_ring_va: 0x0000200000007000",
    "compute_ring_ptb_index: 7",
    "compute_rptr_wptr_timeline_va: 0x000020000000f000",
    "compute_rptr_wptr_timeline_ptb_index: 15",
    "compute_eop_va: 0x0000200000010000",
    "compute_eop_ptb_index: 16",
    "status: pass",
)

EXPECTED_AM_VM_TLB_SEQUENCE_LINES = (
    "self_test: am-vm-tlb-sequence",
    "vmid: 0",
    "flush_order: hdp,mm,mm_reserved_cid2,gc",
    "invalidate_mask: 0x00000001",
    "invalidate_l2_ptes: 1",
    "invalidate_l2_pde0: 1",
    "invalidate_l2_pde1: 1",
    "invalidate_l2_pde2: 1",
    "invalidate_l1_ptes: 1",
    "clear_protection_fault_status_addr: 0",
    "mm_waits: sem,ack",
    "gc_waits: sem,ack",
    "status: pass",
)

EXPECTED_SDMA_RING_SETUP_LINES = (
    "self_test: sdma-ring-setup",
    "sdma_ip_hw_id: 42",
    "sdma_ip_version: 7.0.1",
    "queue_index: 0",
    "register_prefix: regSDMA0_QUEUE0",
    "register_instance: 0",
    "teardown_order: disable_rb,disable_ib,disable_doorbell,clear_doorbell_offset,soft_reset_sdma0",
    "soft_reset_sdma0_bit: 23",
    "ring_va: 0x0000200000003000",
    "ring_size_bytes: 2048",
    "ring_size_field: 9",
    "rptr_va: 0x0000200000003800",
    "wptr_va: 0x0000200000003808",
    "fence_va: 0x0000200000003810",
    "doorbell_index: 256",
    "doorbell_offset_field: 512",
    "doorbell_bar2_byte_offset: 0x0000000000000800",
    "status: pass",
)

EXPECTED_SDMA_FENCE_PACKET_ENCODING_LINES = (
    "self_test: sdma-fence-packet-encoding",
    "packet_dword_count: 4",
    "fence_value: 1",
    "fence_address: 0x0000200000003810",
    "fence_address_le: 1038000000200000",
    "packet_hex: 05000300103800000020000001000000",
    "status: pass",
)

EXPECTED_SDMA_SUBMIT_SEQUENCE_LINES = (
    "self_test: sdma-submit-sequence",
    "copy_packet_dwords: 7",
    "fence_packet_dwords: 4",
    "submit_copy_count: 2",
    "submit_dword_count: 18",
    "initial_wptr_bytes: 0",
    "final_wptr_bytes: 72",
    "doorbell_value: 72",
    "status: pass",
)

EXPECTED_KERNEL_PROOF_CONTRACT_LINES = (
    "self_test: kernel-proof-contract",
    "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
    "kernel_proof_mode: minimal-u32-add-one",
    "kernel_arch: gfx1201",
    "element_type: uint32_t",
    "element_count: 8",
    "input_byte_count: 32",
    "output_byte_count: 32",
    "input_values_u32: 1,2,3,4,5,6,7,8",
    "input_bytes_hex: 0100000002000000030000000400000005000000060000000700000008000000",
    "expected_output_values_u32: 2,3,4,5,6,7,8,9",
    "expected_output_bytes_hex: 0200000003000000040000000500000006000000070000000800000009000000",
    "expected_output_sha256: b06e51b2494d439f5e151692ca393efc3c52cdfddcc377be789356250b9860a6",
    "kernel_source_id: c0a-minimal-u32-add-one-v3",
    "kernel_source_language: amd-gcn-assembly",
    "kernel_blob_format: amdgpu-code-object-v5",
    "kernel_blob_symbol: c0a_minimal_u32_add_one",
    "kernel_blob_target: gfx1201",
    "kernel_text_provenance_path: .superpowers/swarm/reports/c0a-compute-task-5-dispatch.md#task-set-2-kernel-text-provenance",
    "kernel_blob_reference_hsaco_sha256: 7e03c75bb6682d0bb7e688a409c5f53a20a1b3a60b53c7720706500c4e7ae8bf",
    "kernel_blob_reference_text_sha256: 08fd705ca25c7a1d5531e504eb9905ce84dab9c0a31b7ef6ecfc62475b98f965",
    "kernel_blob_reference_text_byte_count: 64",
    "kernel_text_first64_hex: 004100f4000000f8000000f4180000f80000c7bf820002308002047e060005ee03000000010000000000c0bf0006064a048006ee00008001010000000000b0bf",
    "kernel_text_last16_hex: 048006ee00008001010000000000b0bf",
    "kernel_descriptor_kernarg_size: 24",
    "compute_ring_gpu_va: 0x0000200000007000",
    "compute_ring_size_bytes: 32768",
    "compute_rptr_gpu_va: 0x000020000000f000",
    "compute_wptr_gpu_va: 0x000020000000f008",
    "compute_timeline_gpu_va: 0x000020000000f010",
    "compute_eop_gpu_va: 0x0000200000010000",
    "compute_doorbell_index: 3",
    "compute_doorbell_bar2_byte_offset: 0x0000000000000018",
    "kernel_blob_load_status: not_run_no_hardware_contract",
    "kernarg_write_status: not_run_no_hardware_contract",
    "sdma_h2d_status: not_run_no_hardware_contract",
    "sdma_d2h_status: not_run_no_hardware_contract",
    "compute_ring_setup_status: not_run",
    "compute_hqd_active_status: not_run",
    "mec_rs64_cntl_write_status: not_run",
    "mec_rs64_cntl_readback: not_run",
    "mec_rs64_active_status: not_run",
    "kernel_launch_status: not_run_no_hardware_contract",
    "kernel_elapsed_usec: 0",
    "cpu_comparison_status: pass",
    "host_device_transfer_status: not_run_no_hardware_contract",
    "failure_stage: none",
    "failure_text: none",
    "exit_status: 0",
)

EXPECTED_COMPUTE_VM_LAYOUT_LINES = (
    "self_test: compute-vm-layout",
    "kernel_input_vram_va: 0x0000200000001000",
    "kernel_output_vram_va: 0x0000200000004000",
    "kernel_code_vram_va: 0x0000200000005000",
    "kernel_kernargs_va: 0x0000200000006000",
    "compute_ring_va: 0x0000200000007000",
    "compute_rptr_va: 0x000020000000f000",
    "compute_wptr_va: 0x000020000000f008",
    "compute_timeline_va: 0x000020000000f010",
    "compute_eop_va: 0x0000200000010000",
    "compute_control_requested_size: 40960",
    "compute_control_queue_cpu_offset: 0",
    "compute_control_kernargs_cpu_offset: 4096",
    "kernel_input_ptb_index: 1",
    "kernel_output_ptb_index: 4",
    "kernel_code_ptb_index: 5",
    "kernel_kernargs_ptb_index: 6",
    "compute_ring_ptb_index: 7",
    "compute_rptr_wptr_timeline_ptb_index: 15",
    "compute_eop_ptb_index: 16",
    "compute_mqd_paddr: 0x0000000002003000",
    "compute_ring_page_count: 8",
    "status: pass",
)

EXPECTED_GFX_RING_REGISTER_LINES = (
    "self_test: gfx-ring-registers",
    "gc_ip_version: 12.0.1",
    "direct_pm4_requires_xcc_count: 1",
    "mec_doorbell_index: 3",
    "mec_doorbell_bar2_byte_offset: 0x0000000000000018",
    "grbm_select_reg: regGRBM_GFX_INDEX",
    "hqd_reg_span: regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI",
    "compute_set_sh_base: 0x00002c00",
    "compute_pgm_lo_set_sh_offset: 0x0000020c",
    "compute_user_data_0_set_sh_offset: 0x00000240",
    "status: pass",
)

EXPECTED_COMPUTE_MQD_ENCODING_LINES = (
    "self_test: compute-mqd-encoding",
    "mqd_size_bytes: 2048",
    "mqd_dword_count: 512",
    "mqd_hqd_register_copy_start: 128",
    "mqd_cp_mqd_control_span_index: 34",
    "mqd_cp_hqd_eop_base_addr_span_index: 37",
    "mqd_cp_hqd_eop_control_span_index: 39",
    "mqd_cp_mqd_base_addr_lo: 0x02003000",
    "mqd_cp_mqd_base_addr_hi: 0x00000080",
    "mqd_cp_hqd_pq_base_lo: 0x00000070",
    "mqd_cp_hqd_pq_base_hi: 0x00000020",
    "mqd_cp_hqd_pq_wptr_poll_addr_lo: 0x0000f008",
    "mqd_cp_hqd_pq_wptr_poll_addr_hi: 0x00002000",
    "mqd_cp_hqd_eop_base_addr_lo: 0x00000100",
    "mqd_cp_hqd_eop_base_addr_hi: 0x00000020",
    "mqd_cp_hqd_hq_status0: 0x20004000",
    "mqd_compute_user_data_0: 0x00006000",
    "mqd_header: 0xc0310800",
    "hqd_pipe_priority: 0x00000002",
    "hqd_queue_priority: 0x0000000f",
    "hqd_quantum: 0x00000111",
    "hqd_persistent_state: 0x00005501",
    "hqd_vmid: 0",
    "hqd_aql_control: 0",
    "hqd_pq_control_mode: direct_pm4",
    "hqd_copy_expect_cp_hqd_pq_control: 0x0000050c",
    "hqd_pq_doorbell_control: 0x40000018",
    "hqd_ib_control: 0x00300000",
    "hqd_eop_control: 0x00000009",
    "cp_mqd_control: 0x00000100",
    "compute_static_thread_mgmt: 0xffffffff",
    "status: pass",
)

EXPECTED_PM4_DISPATCH_SEQUENCE_LINES = (
    "self_test: pm4-dispatch-sequence",
    "packet_order: acquire_mem,set_sh_pgm,set_sh_rsrc,set_sh_rsrc3,set_sh_tmpring,set_sh_restart,set_sh_userdata,set_sh_resource_limits,set_sh_start,dispatch_direct,event_write,release_mem",
    "packet_count: 12",
    "dispatch_dword_count: 59",
    "compute_wptr_unit: dwords",
    "compute_doorbell_value: 59",
    "packet3_acquire_mem: 0x58",
    "packet3_set_sh_reg: 0x76",
    "packet3_dispatch_direct: 0x15",
    "packet3_event_write: 0x46",
    "packet3_release_mem: 0x49",
    "global_size_x: 1",
    "global_size_y: 1",
    "global_size_z: 1",
    "local_size_x: 8",
    "local_size_y: 1",
    "local_size_z: 1",
    "dispatch_initiator: 0x00000005",
    "release_mem_timeline_value: 1",
    "status: pass",
)

EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES = (
    "self_test: compute-doorbell-delivery",
    "diagnostic_contract: mec_doorbell_delivery_ring_fetch",
    "failure_stage_if_timeline_timeout: kernel_timeline_timeout",
    "classification_if_not_consumed: compute_doorbell_not_consumed",
    "doorbell_bar: BAR2",
    "doorbell_index: 3",
    "doorbell_byte_offset: 0x0000000000000018",
    "doorbell_value_unit: dwords",
    "doorbell_value_source: pm4_dispatch_dword_count",
    "doorbell_hit_source: regCP_HQD_PQ_DOORBELL_CONTROL.doorbell_hit",
    "pre_ring_reads: regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT,regCP_MEC_DOORBELL_RANGE_LOWER,regCP_MEC_DOORBELL_RANGE_UPPER",
    "post_ring_reads: regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT",
    "timeout_reads: timeline,rptr,wptr,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_STAT",
    "classification_if_rptr_zero_cp_idle: compute_doorbell_not_consumed",
    "classification_if_doorbell_hit_rptr_zero: hqd_ring_fetch_not_started",
    "classification_if_rptr_advances_timeline_zero: pm4_dispatch_or_release_mem_blocked",
    "compute_doorbell_route_readback_field: compute_doorbell_route_readback",
    "compute_doorbell_route_classification_field: compute_doorbell_route_classification",
    "route_readback_registers: regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN,regRCC_DEV0_EPF2_STRAP2,regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL,regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL",
    "route_expected_entry0_ctrl: 0x30000007",
    "route_expected_entry3_ctrl: 0x3000000d",
    "route_classification_values: gdc_s2a_route_readback_matches,gdc_s2a_route_readback_mismatch,gdc_s2a_route_readback_unclassified",
    "status: pass",
)

EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES = (
    "self_test: compute-doorbell-consumption",
    "diagnostic_contract: hqd_pq_doorbell_consumption",
    "source_gap_exit_required: diagnostic_override_allowed",
    "hqd_doorbell_control_reads: regCP_HQD_PQ_DOORBELL_CONTROL",
    "hqd_doorbell_control_decodes: doorbell_mode,doorbell_bif_drop,doorbell_offset,doorbell_source,doorbell_schd_hit,doorbell_en,doorbell_hit",
    "hqd_doorbell_control_mqd_compare_ignored_bits: doorbell_bif_drop,doorbell_schd_hit,doorbell_hit",
    "expected_doorbell_offset: 6",
    "expected_doorbell_en: 1",
    "mqd_hqd_compare_fields: cp_hqd_pq_doorbell_control,cp_hqd_pq_control,cp_hqd_pq_base,cp_hqd_pq_rptr_report_addr,cp_hqd_pq_wptr_poll_addr,cp_mqd_control,cp_hqd_eop_base_addr,cp_hqd_eop_control",
    "wptr_visibility_reads: control_wptr_cpu,control_rptr_cpu,regCP_HQD_PQ_WPTR_LO,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_RPTR",
    "cp_mec_status_reads: regCP_STAT,regCP_INT_CNTL_RING0,regCP_MEC1_F32_INTERRUPT,regCP_MEC1_INSTR_PNTR,regCP_MEC_RS64_INTERRUPT,regCP_MEC_RS64_PENDING_INTERRUPT,regCP_MEC_RS64_EXCEPTION_STATUS",
    "cp_mec_rs64_context_reads: regCP_MEC_RS64_INSTR_PNTR,regCP_MEC_RS64_PRGRM_CNTR_START_HI,regCP_MEC_LOCAL_INSTR_BASE_LO,regCP_MEC_LOCAL_INSTR_BASE_HI,regCP_MEC_LOCAL_INSTR_MASK_LO,regCP_MEC_LOCAL_INSTR_MASK_HI,regCP_MEC_LOCAL_INSTR_APERTURE,regCP_MEC_RS64_INTERRUPT_DATA_16,regCP_MEC_RS64_INTERRUPT_DATA_17,regCP_MEC_RS64_INTERRUPT_DATA_18,regCP_MEC_RS64_INTERRUPT_DATA_19,regCP_MEC_RS64_INTERRUPT_DATA_20,regCP_MEC_RS64_INTERRUPT_DATA_21,regCP_MEC_RS64_INTERRUPT_DATA_22,regCP_MEC_RS64_INTERRUPT_DATA_23,regCP_MEC_RS64_INTERRUPT_DATA_24,regCP_MEC_RS64_INTERRUPT_DATA_25,regCP_MEC_RS64_INTERRUPT_DATA_26,regCP_MEC_RS64_INTERRUPT_DATA_27,regCP_MEC_RS64_INTERRUPT_DATA_28,regCP_MEC_RS64_INTERRUPT_DATA_29,regCP_MEC_RS64_INTERRUPT_DATA_30,regCP_MEC_RS64_INTERRUPT_DATA_31",
    "classification_if_rs64_exception_status_nonzero: rs64_exception_context_needed",
    "classification_if_bif_drop: doorbell_route_or_range_drop",
    "classification_if_schd_or_hit_rptr_zero: hqd_doorbell_seen_ring_fetch_not_started",
    "classification_if_wptr_not_visible: compute_wptr_not_visible_to_cp",
    "classification_if_mqd_hqd_mismatch: mqd_hqd_copy_mismatch",
    "classification_if_rptr_advances_timeline_zero: ring_fetch_started_pm4_or_release_mem_blocked",
    "classification_if_no_signal: doorbell_not_reaching_hqd_unclassified",
    "status: pass",
)

EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_CLASSIFIER_LINES = (
    "self_test: compute-doorbell-consumption-classifier",
    "rs64_exception_status: 0x0000c67a",
    "classification: rs64_exception_context_needed",
    "status: pass",
)


EXPECTED_GC_HUB_SEQUENCE_LINES = (
    "self_test: gc-hub-sequence",
    "topology_requirement: one_gc_instance_for_direct_pm4",
    "gc_context: VMID0",
    "sequence: hdp,gc_system_aperture,gc_l1_l2,gc_context0,gc_identity_aperture,gc_invalidate_ranges,gc_tlb_flush",
    "failure_stage_if_multi_xcc: multi_xcc_aql_required",
    "status: pass",
)

EXPECTED_MEC_RS64_PIPE_ACTIVATION_LINES = (
    "self_test: mec-rs64-pipe-activation",
    "cntl_register: regCP_MEC_RS64_CNTL",
    "cntl_offset: 10500",
    "cntl_segment: 1",
    "mec_grbm_select: ME=1 pipe=0 queue=0",
    "program_counter_low_register: regCP_MEC_RS64_PRGRM_CNTR_START",
    "program_counter_low_offset: 10496",
    "program_counter_high_register: regCP_MEC_RS64_PRGRM_CNTR_START_HI",
    "program_counter_high_offset: 10552",
    "program_counter_segment: 1",
    "program_counter_pair_zero_rejected: true",
    "program_counter_pair_restored_exactly: true",
    "program_counter_pair_readback_matches: true",
    "replay_sequence: select_mec,read_start_low,read_start_high,reject_zero_pair,assert_reset,restore_start_low,restore_start_high,verify_start_low,verify_start_high,activate",
    "mec_pipe0_reset_bit: 0x00010000",
    "mec_pipe0_active_bit: 0x04000000",
    "steady_mask: 0xbff0ffff",
    "sample_prior: 0x04001234",
    "sample_reset_write: 0x04011234",
    "sample_steady_write: 0x04001234",
    "status: pass",
)

EXPECTED_COMPUTE_READBACK_CLASSIFIER_LINES = (
    "self_test: compute-readback-classifier",
    "example_observed_hex: 0000020000000300000004000000050000000000000000000000000000000000",
    "example_expected_hex: 0200000003000000040000000500000006000000070000000800000009000000",
    "anomaly_class: swap_and_partial",
    "written_element_mask: 0x0f",
    "swapped_element_mask: 0x0f",
    "unswapped_match_element_mask: 0x0f",
    "status: pass",
)

EXPECTED_KERNEL_TEXT_DECODE_LINES = (
    "self_test: kernel-text-decode",
    "text_byte_count: 64",
    "store_instruction_count: 1",
    "store_class: global",
    "store_primary_op: GLOBAL_STORE_B32",
    "store_addressing: lane+segment",
    "store_element_bounds: 0..0",
    "load_saddr_pair: s[6:7]",
    "store_saddr_pair: s[4:5]",
    "lane_scale_word_present: true",
    "status: pass",
)


def compile_probe(tmp_path):
    assert PROBE_SOURCE.exists(), "native transfer probe source missing"
    exe = tmp_path / "native_amdev_transfer_probe"
    subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(PROBE_SOURCE),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return exe


def run_self_test(exe, name):
    completed = subprocess.run(
        [str(exe), "--self-test", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_remote_cmd_frame_self_test_passes(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "remote-cmd-frame")

    assert "self_test: remote-cmd-frame" in stdout
    assert "status: pass" in stdout
    assert "frame_size: 33" in stdout
    assert f"frame_hex: {EXPECTED_REMOTE_CMD_FRAME_HEX}" in stdout


def test_log_contract_self_test_lists_required_fields(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "log-contract")

    for field in REQUIRED_LOG_FIELDS:
        assert f"required_log_field: {field}" in stdout
    assert "status: pass" in stdout


def test_sysmem_page_list_self_test_expands_and_truncates(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "sysmem-page-list")

    assert stdout.splitlines() == list(EXPECTED_SYSMEM_PAGE_LIST_LINES)


def test_sdma_packet_encoding_self_test_reports_linear_copy_little_endian_fields(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "sdma-packet-encoding")

    assert stdout.splitlines() == list(EXPECTED_SDMA_PACKET_ENCODING_LINES)


def test_am_vm_pte_encoding_self_test_reports_gfx12_flags(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "am-vm-pte-encoding")

    assert stdout.splitlines() == list(EXPECTED_AM_VM_PTE_ENCODING_LINES)


def test_am_vm_page_table_plan_self_test_reports_fixed_indices(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "am-vm-page-table-plan")

    assert stdout.splitlines() == list(EXPECTED_AM_VM_PAGE_TABLE_PLAN_LINES)


def test_am_vm_tlb_sequence_self_test_reports_vmid0_flush_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "am-vm-tlb-sequence")

    assert stdout.splitlines() == list(EXPECTED_AM_VM_TLB_SEQUENCE_LINES)


def test_sdma_ring_setup_self_test_reports_queue0_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "sdma-ring-setup")

    assert stdout.splitlines() == list(EXPECTED_SDMA_RING_SETUP_LINES)


def test_sdma_fence_packet_encoding_self_test_reports_completion_write(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "sdma-fence-packet-encoding")

    assert stdout.splitlines() == list(EXPECTED_SDMA_FENCE_PACKET_ENCODING_LINES)


def test_sdma_submit_sequence_self_test_reports_ring_write_pointer_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "sdma-submit-sequence")

    assert stdout.splitlines() == list(EXPECTED_SDMA_SUBMIT_SEQUENCE_LINES)


def test_kernel_proof_contract_self_test_reports_minimal_u32_shape(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "kernel-proof-contract")

    assert stdout.splitlines() == list(EXPECTED_KERNEL_PROOF_CONTRACT_LINES)


def test_compute_vm_layout_self_test_reports_fixed_pages(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-vm-layout")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_VM_LAYOUT_LINES)


def test_gfx_ring_registers_self_test_reports_source_grounded_offsets(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "gfx-ring-registers")

    assert stdout.splitlines() == list(EXPECTED_GFX_RING_REGISTER_LINES)


def test_compute_mqd_encoding_self_test_reports_hqd_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-mqd-encoding")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_MQD_ENCODING_LINES)


def test_pm4_dispatch_sequence_self_test_reports_direct_dispatch_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "pm4-dispatch-sequence")

    assert stdout.splitlines() == list(EXPECTED_PM4_DISPATCH_SEQUENCE_LINES)

def test_compute_doorbell_delivery_self_test_reports_diagnostic_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-doorbell-delivery")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES)

def test_compute_doorbell_consumption_self_test_reports_hqd_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-doorbell-consumption")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_LINES)

def test_compute_doorbell_consumption_classifier_self_test_reports_rs64_exception(
    tmp_path,
):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-doorbell-consumption-classifier")

    assert stdout.splitlines() == list(
        EXPECTED_COMPUTE_DOORBELL_CONSUMPTION_CLASSIFIER_LINES
    )



def test_gc_hub_sequence_self_test_reports_direct_pm4_vmid0_contract(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "gc-hub-sequence")

    assert stdout.splitlines() == list(EXPECTED_GC_HUB_SEQUENCE_LINES)


def test_gc_hub_source_contract_disables_agp_and_configures_every_invalidate_range():
    source = PROBE_SOURCE.read_text()
    start = source.index("bool program_gc_hub_vmid0(")
    end = source.index("bool poll_register_mask", start)
    gc_hub = source[start:end]

    agp_base = "regs_gfx1201::kGcMcVmAgpBase, 0U"
    agp_bot = "regs_gfx1201::kGcMcVmAgpBot, 0xffffffU"
    agp_top = "regs_gfx1201::kGcMcVmAgpTop, 0U"
    range_loop = "for (const RegDef& range_lo : regs_gfx1201::kGcInvalidateEngAddrRangeLo)"
    range_write = "write_register_pair(client, *log, log->ip.gc, range_lo,"
    range_value = "0x1fffffffffULL, error_text)"
    context_enable = "regs_gfx1201::kGcContext0Cntl,\n                            encode_context0_cntl()"

    assert all(marker in gc_hub for marker in (agp_base, agp_bot, agp_top, range_loop, range_write, range_value))
    assert "constexpr std::array<RegDef, 18> kGcInvalidateEngAddrRangeLo" in source
    assert "constexpr std::array<RegDef, 18> kGcInvalidateEngAddrRangeHi" in source
    for engine in range(18):
        assert f"regGCVM_INVALIDATE_ENG{engine}_ADDR_RANGE_LO32" in source
        assert f"regGCVM_INVALIDATE_ENG{engine}_ADDR_RANGE_HI32" in source
    assert max(gc_hub.index(marker) for marker in (agp_base, agp_bot, agp_top, range_write)) < gc_hub.index(
        context_enable
    )


def test_probe_gc_tlb_flush_uses_req_ack_without_mmhub_semaphore():
    source = PROBE_SOURCE.read_text()
    gc_start = source.index("bool flush_gc_tlb_vmid0(")
    gc_end = source.index("\n\n\nbool setup_fixed_vm_mapping", gc_start)
    gc_flush = source[gc_start:gc_end]
    mm_start = source.index("bool flush_mmhubs_tlb(")
    mm_flush = source[mm_start:gc_start]

    gc_req = "regs_gfx1201::kGcInvalidateEng17Req"
    gc_ack = "regs_gfx1201::kGcInvalidateEng17Ack"
    mm_sem = "regs_gfx1201::kMmInvalidateEng17Sem"
    mm_req = "regs_gfx1201::kMmInvalidateEng17Req"
    mm_ack = "regs_gfx1201::kMmInvalidateEng17Ack"

    assert "flush_hdp(client, *log, error_text)" in gc_flush
    assert "kGcInvalidateEng17Sem" not in gc_flush
    assert gc_flush.index(gc_req) < gc_flush.index(gc_ack)
    assert mm_flush.index(mm_sem) < mm_flush.index(mm_req) < mm_flush.index(mm_ack) < mm_flush.rindex(mm_sem)


def test_mec_rs64_pipe_activation_self_test_reports_steady_state_encoding(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "mec-rs64-pipe-activation")

    assert stdout.splitlines() == list(EXPECTED_MEC_RS64_PIPE_ACTIVATION_LINES)


def test_compute_readback_classifier_self_test_reports_anomaly(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "compute-readback-classifier")

    assert stdout.splitlines() == list(EXPECTED_COMPUTE_READBACK_CLASSIFIER_LINES)


def test_kernel_text_decode_self_test_reports_store_ops(tmp_path):
    exe = compile_probe(tmp_path)

    stdout = run_self_test(exe, "kernel-text-decode")

    assert stdout.splitlines() == list(EXPECTED_KERNEL_TEXT_DECODE_LINES)


def test_help_lists_hardware_modes(tmp_path):
    exe = compile_probe(tmp_path)

    completed = subprocess.run(
        [str(exe), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--self-test remote-cmd-frame" in completed.stdout
    assert "--self-test log-contract" in completed.stdout
    assert "--self-test sysmem-page-list" in completed.stdout
    assert "--self-test sdma-packet-encoding" in completed.stdout
    assert "--self-test am-vm-pte-encoding" in completed.stdout
    assert "--self-test am-vm-page-table-plan" in completed.stdout
    assert "--self-test am-vm-tlb-sequence" in completed.stdout
    assert "--self-test sdma-ring-setup" in completed.stdout
    assert "--self-test sdma-fence-packet-encoding" in completed.stdout
    assert "--self-test sdma-submit-sequence" in completed.stdout
    assert "--self-test kernel-proof-contract" in completed.stdout
    assert "--self-test compute-vm-layout" in completed.stdout
    assert "--self-test gfx-ring-registers" in completed.stdout
    assert "--self-test compute-mqd-encoding" in completed.stdout
    assert "--self-test pm4-dispatch-sequence" in completed.stdout
    assert "--self-test compute-doorbell-delivery" in completed.stdout
    assert "--self-test compute-doorbell-consumption" in completed.stdout
    assert "--self-test compute-doorbell-consumption-classifier" in completed.stdout
    assert "--self-test gc-hub-sequence" in completed.stdout
    assert "--self-test mec-rs64-pipe-activation" in completed.stdout
    assert "--self-test compute-readback-classifier" in completed.stdout
    assert "--self-test kernel-text-decode" in completed.stdout
    assert "--discovery-smoke" in completed.stdout
    assert "--transfer-proof" in completed.stdout
    assert "--kernel-proof" in completed.stdout
