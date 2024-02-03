# Review Document for Data Structures and Algorithms
# To-Do
[ ] add ipynb for current solutions
[ ] update ipynb links for current current solutions

### Resources
Links:
- Rosa Session: https://chat.openai.com/g/g-6VqZewHmW-rosagpt/c/525051e5-9c63-4d34-8fc1-9a87015fa474
- Anki: https://ankiweb.net/decks
- LeetCode: https://leetcode.com/
- concept map: https://roadmap.sh/computer-science
- neetcode: https://neetcode.io/courses/dsa-for-beginners/1

# Introduction
This is a master document that I'm creating while reviewing the fundamentals for algorithm and coding problem solving. The document closely follows the review heirarchy structure of [neetcode](https://neetcode.io), created by Navdeep Singh. The document contains overview sections of data structures and algorithms as well as associated LeetCode problems that can be used to reinforce the topic concepts.

# Data Structures
## Basic
### Arrays
Arrays are data structures used to store an ordered collection of items in *contiguous* (consecutive) memory. Arrays can be static, where the size is defined at compile time, or dynamic, where the size can change during runtime. Arrays can be multi-dimensional. Data in arrays are accessed via indexing (e.g., Python zero-based indexing). At the memory level, information for each element is calculated using the base address and the size of the array data types (e.g., int8).

Because of the fixed data type and sizing constraints, arrays are highly effecient and performant data structures.

#### Problems
26. Remove Duplicates from Sorted Array ([Problem link](https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/); [Solution link](data_structures/arrays/00_lc26_duplicates.ipynb))
27. Remove Element ([Problem Link](https://leetcode.com/problems/remove-element/); [Solution Link](data_structures/arrays/01_lc27_remove_element.ipynb))
1929. Concatenation of Array ([Problem Link](https://leetcode.com/problems/concatenation-of-array/); [Solution Link](data_structures/arrays/02_lc1929_concatenation.ipynb))

### Stacks
Stacks are an abstract data type that act as collections of objects. They can
be constructed with arrays or linked lists.

Stacks follow the *LIFO* principle (Last In, First Out) and have two main
operations:
- Push: Add an element to top of the stack
- Pop: Remove the top element from the stack

The tops of stacks can also be read without popping.

Stacks are useful for function call management, undo functions, and depth-first
search (DFS) graph algorithms.

#### Problems
682. Baseball Game ([Problem Link](https://leetcode.com/problems/baseball-game/); [Solution Link](data_structures/01_stacks/03_lc682_baseball_game.ipynb))
20. Valid Parentheses ([Problem Link](https://leetcode.com/problems/valid-parentheses); [Solution Link](data_structures/01_stacks/04_lc20_valid_parentheses.ipynb))
155. Min Stack ([Problem Link](https://leetcode.com/problems/min-stack/); [Solution Link](data_structures/01_stacks/05_lc155_min_stack.ipynb))

### Linked Lists
Linked lists are data structures with non-contiguous and sequential data storage. Because are not stored contiguously, they are memory efficient and can be dynamically resized. They are useful for contexts that require frequent
insertions/deletions and are sufficient for sequential access operations. Linked
lists are the fundamental data structure for stacks, queues, and some hash
tables.

Linked lists are composed of nodes, which are elements that contain two separate parts of information:
- **Value** - The value stored in the node
- **Pointer** - The point in that points to other nodes in the list.

Linked lists can have multiple structures:
- **Singly Linked** - Each node only points to one other node. The end node 
points to nothing.
- **Doubly Linked** - Each node points to the node before and after it in the
stack.
- **Circular** - The linked list forms a closed loop.

#### Problems
206. Reverse Linked List ([Problem Link](https://leetcode.com/problems/reverse-linked-list/); [Solution Link](data_structures/02_linked_lists/06_lc206_reverse_linked_list.ipynb))
21. Merge Two Sorted Lists ([Problem Link](https://leetcode.com/problems/merge-two-sorted-lists/); [Solution Link](data_structures/02_linked_lists/07_lc21_merged_sorted_lists.ipynb))
707. Design Browser History ([Problem Link](https://leetcode.com/problems/design-browser-history/description/); [Solution Link](data_structures/02_linked_lists/08_lc1472_design_browser_history.ipynb))

### Hash Tables
Understanding hashing, collision resolution techniques, implementation of maps and sets

## Advanced
### Trees
Binary trees, binary search trees, AVL trees, Red-Black trees, segment trees, Fenwick trees (Binary Indexed Trees), trie (prefix tree).

### Graphs
Representation (adjacency matrix, adjacency list), traversal algorithms (BFS, DFS), topological sorting.

### Heaps
Min-heap, max-heap, priority queues.

### Disjoint Set Union

### Advanced Graph Structures
Directed Acyclic Graphs (DAGs), minimum spanning trees, shortest path algorithms (Dijkstra’s, Bellman-Ford, Floyd-Warshall).

# Algorithms
## Basic
### Sorting
Bubble sort, insertion sort, selection sort, merge sort, quicksort, heapsort

### Searching
Linear search, binary search.

### Recursive
Understanding recursion, examples like factorial, Fibonacci series.

## Advanced
### Dynamic Programming
Understanding overlapping subproblems and optimal substructure, memoization, tabulation

### Greedy
Understanding the greedy approach, problems like activity selection, fractional knapsack

### Graph
Advanced traversal techniques, shortest path problems (Dijkstra’s, Bellman-Ford, Floyd-Warshall), network flow problems (like Ford-Fulkerson).

### Backtracking
Solving combinatorial problems, understanding pruning in backtracking

### Divide and Conquer
Understanding the divide-and-conquer strategy, problems like quicksort, mergesort, binary search.

### Bit Manipulation
Techniques for efficient manipulation of bits.
