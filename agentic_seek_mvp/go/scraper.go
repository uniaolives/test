// go/scraper.go
package main

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"time"
)

type TaskRequest struct {
	Action string `json:"action"`
	Target string `json:"target"`
}

type TaskResponse struct {
	Status  string `json:"status"`
	Content string `json:"content"`
}

func main() {
	socketPath := "/tmp/atlas_go.sock"
	os.Remove(socketPath) // Limpa socket antigo

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		panic("Erro ao iniciar socket Go: " + err.Error())
	}
	defer listener.Close()

	fmt.Println("🟢 [GO] Worker AgenticSeek pronto.")

	for {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}

		go handleTask(conn)
	}
}

func handleTask(conn net.Conn) {
	defer conn.Close()

	decoder := json.NewDecoder(conn)
	var req TaskRequest

	if err := decoder.Decode(&req); err != nil {
		fmt.Println("Erro ao decodificar JSON:", err)
		return
	}

	fmt.Printf("🌐 [GO] Executando: %s em %s\n", req.Action, req.Target)

	// Simula busca (HTTP Request)
	time.Sleep(500 * time.Millisecond)

	// Resposta
	resp := TaskResponse{
		Status:  "success",
		Content: fmt.Sprintf("Dados coletados de %s", req.Target),
	}

	json.NewEncoder(conn).Encode(resp)
	fmt.Println("✅ [GO] Tarefa concluída.")
}
