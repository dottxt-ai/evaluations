import json
import outlines

# These are the standard 8 prompts used in the
# lm-evaluation-harness.
qa_8 = [
    {
        "question":  "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "reasoning":  "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        "answer": 6
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "answer": 5
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "answer": 39
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "answer": 8
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "reasoning": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
        "answer": 9
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "reasoning": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
        "answer": 29
    },
    {
        "question":  "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "reasoning": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
        "answer": 33
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "reasoning": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
        "answer": 8
    },   
]

def standard_prompter(n_shot=8, cot=True, examples=qa_8):
    if cot:
      base_prompt = "\n\n".join([
          f"Q: {ex['question']}\nA: {ex['reasoning']} The answer is {ex['answer']}." 
          for ex in examples[0:n_shot]
      ])
    else:
      base_prompt = "\n\n".join([
          f"Q: {ex['question']}\nA: The answer is {ex['answer']}." 
          for ex in examples[0:n_shot]
      ])
    def prompter(question):
        # for consistency, keep this but the A should be added
        # return base_prompt + f"\n\nQ: {question}\nA:"
       return base_prompt + f"\n\nQ: {question}\n"
    return prompter

def alt_1_prompter(n_shot=8, cot=True, examples=qa_8):
    if cot:
      base_prompt = "\n\n".join([
          f"Question - {ex['question']}\nAnswer - {ex['reasoning']} The answer is {ex['answer']}." 
          for ex in examples[0:n_shot]
      ])
    else:
      base_prompt = "\n\n".join([
          f"Question - {ex['question']}\nAnswer - The answer is {ex['answer']}." 
          for ex in examples[0:n_shot]
      ])
    def prompter(question):
        # for consistency, keep this but the A should be added
        # return base_prompt + f"\n\nQ: {question}\nA:"
       return base_prompt + f"\n\nQuestion - {question}\n"
    return prompter   

def json_hr_prompter(n_shot=8, cot=True, examples=qa_8):
  if cot:
      base_prompt = ",\n".join([
"""{{
 "question": "{0}",
 "response": {{
    "reasoning": "{1}",
    "answer": {2}
   }},
}}
""".format(ex['question'],ex['reasoning'], ex['answer']) for ex in examples
      ])
  else:
    base_prompt = ",\n".join([
       """{{
 "question": "{0}",
 "response": {{
    "answer": {1}
   }},
}}
""".format(ex['question'], ex['answer']) for ex in examples
     ])
  def prompter(question):
     return base_prompt + """{0}
  "question": "{1}",
  "response":
  """.format("{",question)
  return prompter


# keep this around until you've confirmed it's identical
def standard_8(question):
    return """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\n\
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\n\
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\n\
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\n\
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\n\
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\n\
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\n\
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n\
Q: {question}\n""".format(question=question)




@outlines.prompt
def json_hr_8(question):
    """{
 "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
 "response": {
    "reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
    "answer": 6
   },
},
{
 "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
 "response": {
    "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
    "answer": 5
   }
},
{
  "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
  "response": {
    "reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
    "answer": 39
   }
},
{
  "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
  "response": {
    "reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
    "answer": 8
   }
},
{
  "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
  "response": {
    "reasoning": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
    "answer": 9
   }
},
{
  "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
  "response": {
    "reasoning": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
    "answer": 29
   }
},
{
  "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
  "response": {
    "reasoning": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
    "answer": 33
   }
},
{
  "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
  "response": {
    "reasoning": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
    "answer": 8
   }
},
{
  "question": "{{question}}",
  "response":
    """

def json_m_8(question):
    examples_dicts=[
        {
            "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
             "response": {
               "reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
               "answer": 6
          }
        },
        {
         "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
         "response": {
           "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
           "answer": 5
          }
        },
        {
          "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
          "response": {
            "reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
            "answer": 39
           }
        },
        {
          "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
          "response": {
            "reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
            "answer": 8
           }
        },
        {
          "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
          "response": {
            "reasoning": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
            "answer": 9
           }
        },
        {
          "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
          "response": {
            "reasoning": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
            "answer": 29
           }
        },
        {
          "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
          "response": {
            "reasoning": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
            "answer": 33
           }
        },
        {
          "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
          "response": {
            "reasoning": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
            "answer": 8
           }
        },
    {"question": f"{question}",
     "response": {}}
    ]
    text_json = json.dumps(examples_dicts)
    return text_json.strip("[]}")[0:-1]

prompt_map = { 
    'standard': standard_prompter,
    'alt_1': alt_1_prompter,
    'json_hr': json_hr_prompter,
    # legacy
    'json_hr_8': lambda *args: json_hr_8,
    'json_m_8': lambda *args: json_m_8
}
