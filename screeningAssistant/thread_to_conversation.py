import openai

thread_id = "thread_EIpHrTAVe0OzoLQg3TXfvrkG"

messages = []

for page in openai.beta.threads.messages.list(thread_id=thread_id, order="asc").iter_pages():
    messages += page.data

items = []
for m in messages:
    item = {"role": m.role}
    item_content = []

    for content in m.content:
        match content.type:
            case "text":
                item_content_type = "input_text" if m.role == "user" else "output_text"
                item_content += [{"type": item_content_type, "text": content.text.value}]

    item |= {"content": item_content}
    items.append(item)

# create a conversation with your converted items
conversation = openai.conversations.create(items=items)
print(conversation.id)