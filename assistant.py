def create_vector_store(openai_client, file_paths):
    # Create a vector store called "Chat to PDFs"
    vector_store = openai_client.beta.vector_stores.create(name="Chat to PDFs")

    # Ready the files for upload to OpenAI
    file_streams = [open(path, "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = openai_client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    return vector_store.id


def create_thread(openai_client, user_question):
    # Create a thread and attach the file to the message
    thread = openai_client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Hi, I need help analyzing a few documents to answer some questions. Can you assist?",
            },
            {
                "role": "assistant",
                "content": "Thanks! I will now use the documents to answer your questions. What is your question?"
            },
            {
                "role": "user",
                "content": f"{user_question}"
            }
        ]
    )

    return thread.id
