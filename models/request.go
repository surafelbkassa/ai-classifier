package models

type MessageRequest struct {
	Text string `json:"text"`
}

type MessageResponse struct {
	Text     string `json:"text"`
	Category string `json:"category"`

	Priority string `json:"priority"`
}
