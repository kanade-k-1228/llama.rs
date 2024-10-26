<div align="center">

# llama.rs

**Very simple rust implimentation of LLaMA**

English | [日本語](README_JP.md)

</div>

```bash
$ git clone git@github.com:kanade-k-1228/llama.rs.git
$ cd llama.rs
$ cargo build --release
$ ./target/release/llama -p "Hello my name is"
```

## Run on Raspberry Pi

Install cross compiler for aarch64.

```bash
$ sudo apt update
$ sudo apt install gcc-aarch64-linux-gnu
```

Build binary for aarch64.

```bash
$ cargo build --release --target aarch64-unknown-linux-gnu
```

Send binary & model file to device.

```bash
$ scp ./target/aarch64-unknown-linux-gnu/release/llama pi@?.?.?.?:/home/pi
$ scp -r model pi@?.?.?.?:/home/pi
```

Run on RaspberryPi.

```bash
$ ssh pi@?.?.?.?
$ ./llama
```

## TODO

- Quantization
- use all CPU (4 core)
- use GPU (Broadcom VideoCore)
