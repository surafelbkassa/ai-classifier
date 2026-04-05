package main

import (
	"encoding/json"
	"log"
	"net/http"

	classifier "github.com/surafelbkassa/ai-classifier/classifer"
	"github.com/surafelbkassa/ai-classifier/models"
)

func classifyHandler(w http.ResponseWriter, r *http.Request) {
	var req models.MessageRequest
	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	category, priority := classifier.Classify(req.Text)
	resp := models.MessageResponse{
		Text:     req.Text,
		Category: category,
		Priority: priority,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func main() {
	http.HandleFunc("/classify", classifyHandler)
	log.Println("Server is running on port 8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
