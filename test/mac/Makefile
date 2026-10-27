# Standalone test of the multi-cycle BF16 MAC against test/bf16_model.py
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = $(PWD)/../../src/compute/bf16_mac_seq.sv
COMPILE_ARGS += -g2012
COCOTB_TOPLEVEL = bf16_mac_seq
COCOTB_TEST_MODULES = test_mac
export PYTHONPATH := $(PWD)/..:$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
