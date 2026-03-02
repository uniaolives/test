module sasc.io/sopa/network

go 1.24.3

replace sasc.io/chain1337 => ./chain1337

require (
	github.com/zeebo/blake3 v0.2.4
	sasc.io/chain1337 v0.0.0-00010101000000-000000000000
)

require github.com/klauspost/cpuid/v2 v2.0.12 // indirect
