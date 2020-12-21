all:bench

bench:
	futhark bench hist-clas.fut --backend=cuda
	futhark bench hist-reg.fut --backend=cuda

clean:
	find . -maxdepth 1 -type f -executable -exec rm '{}' \;
	rm *.c
