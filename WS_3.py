#q1
'''def capitals(fruits):
    capitalized_fruits = [i.capitalize() for i in fruits]
    print(capitalized_fruits)
def main():
    fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarian orange']
    capitals(fruits)
if __name__=="__main__":
    main()'''
#2
'''def two_vowels(fruits):
    for item in fruits:
        fruits_with_only_two_vowels = [i for i in fruits if i.count('a') + i.count ('e') + i.count ('i') + i.count ('o') + i.count ('u') == 2]
    print(fruits_with_only_two_vowels)
def main():
    fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
    two_vowels(fruits)

if __name__=="__main__":
    main()
'''

#q4
'''def dic(numbers):
    even_sq = {i: i ** 2 for i in numbers if i % 2 == 0 }
    print(even_sq)
def main():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dic(numbers)
if __name__=="__main__":
    main()'''

#q5
'''def rev_dic(sentence):
        tokens = sentence.split(' ')
        reverse = {item: item[::-1] for item in tokens}
        print(reverse)
def main():
    sentence = "Hello, how are you?"
    rev_dic(sentence)
if __name__=="__main__":
    main()'''

#6
'''a = ["asmita", "mukherjee", "my", "hi", "they"]
y=sorted(a,key=lambda i : i[::-1])
print(y)'''

#7
'''numbers = [-8, 7, -1, 55, -10, 4, 2, 10]
sort_list = lambda numbers: sorted(numbers)
a = sort_list(numbers)
print(a)
'''

#8
'''class logging:
    def log_func(func):
        def abc (*args, **kwargs):
            print(f"calling {func.__name__} with args {args}, kwargs {kwargs}")
            result = func(*args, **kwargs)
            print(f"{func.__name__} returned {result}")
            return result
        return abc
    @log_func
    def add(a,b):
        return a+b
    result = add(2,3)
    print("result")
'''

#10
'''def division(x,y):
    try:
        print(x/y)
    except ZeroDivisionError:
        print("denominator cannot be 0")
    except ValueError:
        print("value error, try again")
    finally:
        print("end")
def main():
    division(20, 0)
if __name__=="__main__":
    main()'''


#11.
'''class FormulaError(Exception):
    pass
def input_check(user_input):
    a = user_input.split()
    if len(a) != 3:
        raise FormulaError("input does not contain 3 elements")
    n1, op, n2 = a
    try:
        n1 = float(n1)
        n2 = float(n2)
    except ValueError:
        raise FormulaError("first and third input should be numbers")
    return n1, op, n2
def calculate(n1, op, n2):
    if op == '+':
        return n1 + n2
    elif op == '-':
        return n1-n2
    elif op == '*':
        return n1*n2
    elif op == '/':
        return n1/n2
    else:
        raise FormulaError('{0} is not a valid operator')
def main():
    while True:
        user_input = input("enter an expression: (or 'quit' to stop): ")
        if user_input == 'quit':
            break
        try:
            n1, op, n2 = input_check(user_input)
            result = calculate(n1, op, n2)
            print(result)
        except FormulaError as e:
            print("error:", e)
if __name__=="__main__":
    main()
'''
