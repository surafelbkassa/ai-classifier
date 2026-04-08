package classifier

import (
	"os/exec"
	"encoding/json"
	"strings"
)

// This calls Python ML model
func Classify(text string) (string, string) {
	// Call Python script
	cmd := exec.Command("python3", "predict.py", text)
	output, err := cmd.Output()
	if err != nil {
		return fallbackClassify(text), "medium"
	}
	
	var result struct {
		Category string `json:"category"`
	}
	json.Unmarshal(output, &result)
	
	priority := map[string]string{
		"spam":     "low",
		"work":     "high",
		"personal": "medium",
	}[result.Category]
	
	return result.Category, priority
}

func fallbackClassify(text string) string {
	text = strings.ToLower(text)
	if strings.Contains(text, "urgent") || strings.Contains(text, "meeting") {
		return "work"
	}
	if strings.Contains(text, "free") || strings.Contains(text, "offer") {
		return "spam"
	}
	return "personal"
}
