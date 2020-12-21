all: bench

bench:
	futhark bench exact-clas.fut --backend=cuda
	futhark bench exact-reg.fut --backend=cuda

clean:
	find . -maxdepth 1 -type f -executable -exec rm '{}' \;
	rm *.c

