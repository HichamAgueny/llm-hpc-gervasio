#!/bin/bash

# Exit immediately if a command fails
set -e

PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/finetuning"
CONNECTION_FILE="${MyWD}/inference/connection.env"

# Wait until the server connection file exists
echo "Waiting for server to start..."
while [[ ! -f "$CONNECTION_FILE" ]]; do
    sleep 5
done

# Load host, port, and model configuration dynamically from the server script
source "$CONNECTION_FILE"

# Fallback mechanism: If $MODEL is empty/not set in connection.env, default to "custom_lora"
TARGET_MODEL="${MODEL:-custom_lora}"

echo "Connected to $HOST:$PORT"
echo "Model: $TARGET_MODEL"
echo "Type 'exit' or 'quit' to end the conversation."
echo "----------------------------------------"

while true; do
    # Read user input safely
    echo -n "You: "
    read -r prompt

    # Graceful exit condition
    if [[ "$prompt" == "exit" || "$prompt" == "quit" ]]; then
        echo "Exiting chat session."
        break
    fi

    # Skip empty lines
    if [[ -z "$prompt" ]]; then
        continue
    fi

    echo -n "A: "

    # 1. Dynamically construct a safe JSON payload using jq with the target model
    # 2. POST to the OpenAI-compatible endpoint
    # 3. Parse out the message content cleanly
    jq -n \
      --arg model "$TARGET_MODEL" \
      --arg prompt "$prompt" \
      '{
        model: $model,
        messages: [{role: "user", content: $prompt}],
        max_tokens: 512
      }' | curl -s -X POST "$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @- | jq -r '.choices[0].message.content // "Error: Empty response or invalid payload received."'

    echo -e "\n---"
done
