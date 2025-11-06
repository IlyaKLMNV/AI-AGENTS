from classifierLLM import ClassifierAssistant
import time


message = """
здравствуйте! спасибо, но я в данный момент такую позицию не рассматриваю)
"""

assistant = ClassifierAssistant()

start_time = time.time()

run = assistant.run(message)
print(run)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
