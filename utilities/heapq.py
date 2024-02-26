# rewrite from the source https://github.com/donsheehy/datastructures.git

class HeapPQEntry:
    def __init__(self, item, priority):
        self.priority = priority
        self.item = item

    def __lt__(self, other):
        return self.priority < other.priority


class HeapPQ:
    def __init__(self):
        self._entries = []

    def _insert(self, item, priority):
        self._entries.append(HeapPQEntry(item, priority))
        self._up_heap(len(self._entries) - 1)

    def _parent(self, i):
        return (i - 1) // 2

    def _leftchild(self, i):
        return 2 * i + 1

    def _rightchild(self, i):
        return 2 * i + 2

    def _children(self, i):
        left = 2 * i + 1
        right = 2 * i + 2
        return range(left, min(len(self._entries), right + 1))

    def _swap(self, a, b):
        elist = self._entries
        elist[a], elist[b] = elist[b], elist[a]

    def _up_heap(self, i):
        elist = self._entries
        parent = self._parent(i)
        if i > 0 and elist[i] < elist[parent]:
            self._swap(i, parent)
            self._up_heap(parent)

    def findmin(self):
        return self._entries[0].item

    def remove_min(self):
        elist = self._entries
        item = elist[0].item
        elist[0] = elist[-1]
        elist.pop()
        self._down_heap(0)
        return item

    def _down_heap(self, i):
        elist = self._entries
        children = self._children(i)
        if children:
            child = min(children, key=lambda x: elist[x])
            if elist[child] < elist[i]:
                self._swap(i, child)
                self._down_heap(child)

    def __len__(self):
        return len(self._entries)

    def _heapify(self):
        n = len(self._entries)
        for i in reversed(range(n)):
            self._down_heap(i)


class PriorityQueue(HeapPQ):
    def __init__(self, entries=()):
        super().__init__()
        self._entries = [HeapPQEntry(i, p) for i, p in entries]
        self._itemmap = {entry.item: index
                         for index, entry in enumerate(self._entries)}
        self._heapify()

    def exists(self, item):
        if item in self._itemmap:
            return True
        else:
            return False

    def insert(self, entry):
        index = len(self._entries)
        self._entries.append(entry)
        self._itemmap[entry.item] = index
        self._up_heap(index)

    def _swap(self, a, b):
        elist = self._entries
        va = elist[a].item
        vb = elist[b].item
        self._itemmap[va] = b
        self._itemmap[vb] = a
        elist[a], elist[b] = elist[b], elist[a]

    def change_priority(self, item, priority):
        i = self._itemmap[item]
        self._entries[i].priority = priority
        # Assuming the tree is heap ordered, only one will have an effect.
        self._up_heap(i)
        self._down_heap(i)

    def _remove_at_index(self, index):
        elist = self._entries
        self._swap(index, len(elist) - 1)
        del self._itemmap[elist[-1].item]
        elist.pop()
        self._down_heap(index)

    def remove_min(self):
        item = self._entries[0].item
        self._remove_at_index(0)
        return item

    def remove(self, item):
        self._remove_at_index(self._itemmap[item])

    def peek_head(self):
        return self._entries[0].item, self._entries[0].priority

    def peek_left_child(self, i):
        left_index = self._leftchild(i)
        return self._entries[left_index].item, self._entries[left_index].priority

    def peek_right_child(self, i):
        right_index = self._rightchild(i)
        return self._entries[right_index].item, self._entries[right_index].priority

    def __iter__(self):
        return self

    def __next__(self):
        if len(self) > 0:
            return self.remove_min()
        else:
            raise StopIteration
