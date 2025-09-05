package main

import (
	"log"

	"internal-perplexity/server/api/server"
)

func main() {
	log.Println("Starting agent server...")

	// Create and start server
	srv, err := server.NewServer(nil) // nil uses default config
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	if err := srv.Start(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
