package main

import (
	"fmt"
	"log"
	"net/http"
)

func main() {
	fmt.Println("=== Arkhe(n) Sync Engine (Ω+197) Ignition ===")
	http.HandleFunc("/sync", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Arkhe Sync Engine is operational")
	})

	log.Fatal(http.ListenAndServe(":8080", nil))
}
