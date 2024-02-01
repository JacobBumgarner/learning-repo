# stacks
# LC 682
# Notes
This problem has O(n) time complexity, as it will always take as long as
the number of input operations Solving the problem with a stack leads to O(n)
space complexity, because the stack will always need to grow with the input ops.
I'm not sure if this problem could be solved with O(1) space complexity, as we
could get a long list of ("C"), which would require knowing an equivalently long
record of the ops.

# Solution Thoughts
This problem should be relatively straight forward to solve using a list that we
grow and shrink according to the input operations. We can operate under the 
assumptions that any removals, doubling, or addition processes will only
appear if there are sufficient values in the stack for the operation.

List operation procedures:
- If we encounter an integer, append it to the list
- If we encounter a removal ("C"), pop the top option (`.pop()`)
- If we encounter a "D", append the previous value times 2, using negative 
indexing (`[-1]`)
- If we encounter a "+", add the previous two values, using negative indexing
(`[-1]`, `[-2]`)

# 682
# start with empty record
# given a list of string operations
# x - Integer to add to the list
# "+" - Record new score that is the sum of the previous two scores
# "D" - Record new score that is double the previous score
# "C" - Remove the previous score from the record
# 
# Return the sum of the record
def x(ops):
  scores = []
  
  for op in ops:
    if op == "+":
      scores.append(scores[-1] + scores[-2])
    elif op == "D": 
      scores.append(scores[-1] * 2)
    elif op == "C":
      scores.pop()
    else:
      scores.append(int(op))
  
  return sum(scores)

ops = ["5", "2", "C", "D", "+"]
print("First:", x(ops))

ops = ["5", "-2", "4", "C", "D", "9", "+", "+"]
print("Second:", x(ops))

ops = ["1", "C"]
print("Third:", x(ops))


# LC 20
# Notes
This problem can be solved using a stack-based approach. As such, the time and
space complexity will be O(n), as the memory requirements and solve times
are linearly dependent on the length of the input string.

# Solution Thoughts
A runnning stack of opening parenthesis will be constructed. Whenever a closing
parenthesis is encountered, it will be compared to the top of the stack. If the
opening parenthesis at the top of the stack matches the closing parenthesis, it
will be popped. If it doesn't, this means that the closing parenthesis has been
added erroneously, so a `False` will be returned.

The end of the function will only be reached if the input `s` is length one, or
if there are no remaining parentheses. Because of this edge case, at the end of
the function, we need to return a bool examining whether the length the running
stack is zero.

# 20
def x(s):
  opening_s = []
  matching_pairs = {")": "(", "]": "[", "}": "{"}
  
  for b in s:
    if b in ["(", "[", "{"]:
      opening_s.append(b)
    elif len(opening_s) and matching_pairs[b] == opening_s[-1]:
      opening_s.pop()
    else: 
      return False
  
  return len(opening_s) == 0

s = "()"
print("First:", x(s))

s = "()[]{}"
print("Second:", x(s))

s = "["
print("Third:", x(s))

# 155. Min Stack
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- `MinStack()` initializes the object
- `push(val)` pushes the element `val` onto the stack
- `pop()` removes and returns the element on the top of the stack
- `top()` gets the top element of the stack
- `getMin()` retrieves the minimum element in the stack

Implement a solution with O(1) time complexity for each function.

# Notes
This problem doesn't really have a space complexity constraint, as we're constructing a class rather than worrying about constructing an algorithm.

The time constraint for this problem indicates that we need to have O(1) complexity for each function. 

This will be straight forward for the `push`, `pop`, and `top` functions. The `getMin` function could easily be O(n) if we searched the entire stack for each call. Instead, perhaps we need to store an ordered list of the current values in the stack. We could keep track of the unique values in the stack and the number of times they appear. This would be an O(len(unique(n)) * 2) space complexity problem.

After a GPT discussion, this approach certainly would work, but there is a simpler O(2n) approach of just keeping a running list of the corresponding minimal value in the stack after each `push` and `pop` operation. This is an example of a time/space trade-off, as the ordered list approach would use less space, but would take more time to implement and would be marginally slower. In addition, both solutions reduce to O(n) space complexity.

I'll follow the running "min-stack" solution for now.

# Solution Notes
I'll opt to not use the built in `pop` operation for this exercise.

The main considerations will be checking that the list has actual values to `top`, `pop`, and `getMin`. `None` will be returned otherwise.

To implement the push function, we will first add the input value to the stack. Then, we need to add conditionals to check how to add the value to the running minimum.

If the stack minimum has no values, we will simply append the input value. Otherwise, we will compare the input value to the top of the stack and append the smallest value to the minimum stack.

# 155
class x(object):
  def __init__(self):
    self.stack = []
    self.stack_min = []
    
    return

  def push(self, val):
    self.stack.append(val)
    
    if not len(self.stack_min):
      self.stack_min.append(val)
    elif val <= self.stack_min[-1]:
      self.stack_min.append(val)
    else:
      self.stack_min.append(self.stack_min[-1])
      
    return
  
  def pop(self):
    if len(self.stack):
      return_value = self.stack[-1]
      del self.stack[-1]
      del self.stack_min[-1]
    else:
      return None
    
    return return_value
  
  def top(self):
    if len(self.stack):
      return_value = self.stack[-1]
    else:
      return None
    
    return return_value
  
  def getMin(self):
    return_value = self.stack_min[-1]
    return return_value
  
a = x()
print(a.push(-2))
print(a.push(0))
print(a.push(-3))
print(a.getMin())
print(a.pop())
print(a.top())
print(a.getMin())

# linked list
# 206. Reverse Linked List
Given the head of a singly linked list, reverse the list, and return the 
reversed list.

# Notes
For this problem, we will return the `head` node of the reversed list, i.e., the
original `end` node.

This problem follows O(n) time complexity, as the solution is linearly
dependent on the size of the input linked list. The problem will follow O(1)
space complexity, as we don't need any dynamically initialized objects to 
reverse the list.

# Solution Thoughts
To reverse the list, we will iterate in a while loop that runs until the 
current node points to `None`.

We will store the previous node, the current node, and the next node.

For each iteration of the loop, we will first grab the `next` node from the
`current` node. We will then point the `current` node to the `previous` node.
Lastly, we will convert the `current` node to the `next` node and restart our
loop.

# 206
def x(head):
  prev_node = None
  current_node = head
  next_node = None
  
  while (current_node is not None):
    next_node = current_node.next
    current_node.next = prev_node
    prev_node = current_node
    current_node = next_node
    
  return prev_node
    
