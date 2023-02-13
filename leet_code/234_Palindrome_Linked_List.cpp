#include <iostream>
using namespace std;

struct ListNode
{
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

int main(ListNode *head)
{
    ListNode *fast = head, *slow = head, *prev, *tmp;

    // reverse the right half of the list
    while (fast && fast->next)
    {
        slow = slow->next;
        fast = fast->next->next;
    }

    prev = NULL;
    while (slow)
    {
        tmp = prev;
        prev = slow;
        slow = slow->next;
        prev->next = tmp;
    }

    // check if palindrome
    slow = prev;
    fast = head;
    while (slow)
    {
        if (slow->val != fast->val)
        {
            return false;
        }
        slow = slow->next;
        fast = fast->next;
    }
    return true;
}

bool too_slow(ListNode *head)
{
    ListNode *right = head;
    int list_length = 0;

    while (right)
    {
        list_length++;
        right = right->next;
    }

    ListNode *left = head;
    for (int i = 0; i < list_length / 2; i++)
    {
        right = left;
        for (int j = 0; j < list_length - (i * 2) - 1; j++)
        {
            right = right->next;
            // cout << right->val << "\n";
        }

        if (left->val != right->val)
        {
            return false;
        }
        // cout << "Left: " << left->val << "; Right: " << right->val << "\n";
        left = left->next;
    }

    return true;
}
