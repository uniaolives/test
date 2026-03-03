package main

func AgentRouter(agentChannel <-chan []byte, memoryChannel chan<- []byte) {
    for thought := range agentChannel {
        go func(t []byte) {
            // Asynchronous semantic broadcast
            memoryChannel <- t
        }(thought)
    }
}
