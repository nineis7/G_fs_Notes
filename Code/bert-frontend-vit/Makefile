script:
	cd tvm/build/ && cmake .. && make -j10

bert:
	rm -rf dump;
	cd tvm/build/ && cmake .. && make -j10
	python3 driver.py 2>&1 | tee driver_bert.log                                                  # run tvm frontend
	python3 utils/run_ir_json.py --config=artifacts/OPU_IR_bert.json --input_sequence_length=64  # run json ir validation

vit:
	rm -rf dump;
	cd tvm/build/ && cmake .. && make -j10
	python3 driver_vit.py 2>&1 | tee driver_vit.log                                               # run tvm frontend
	python3 utils/run_ir_json_vit.py --config=artifacts/OPU_IR_vit.json                           # run json ir validation
