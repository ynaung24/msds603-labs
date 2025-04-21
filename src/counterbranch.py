from metaflow import FlowSpec, step

class CounterBranchFlow(FlowSpec):
    @step
    def start(self):
        self.creature = "dog"
        self.count = 0
        self.next(self.add_one, self.add_two) # branching happens here

    @step
    def add_one(self):
        self.count += 1
        self.next(self.join) # move to the merging step, where the add_one and add_two nodes will merge

    @step
    def add_two(self):
        self.count += 2
        self.next(self.join) # move to the merging step, where the add_one and add_two nodes will merge

    @step 
    def join(self, inputs):
        self.count = max(inp.count for inp in inputs) # iterate over inputs to find max count
        print("count from add_one", inputs.add_one.count) # print specific values from named branch
        print("count from add_two", inputs.add_two.count) # print specific values from named branch

        self.creature = inputs[0].creature # creature was not modified, so we can use the index 0, and this step is required
        self.next(self.end)

    @step
    def end(self):
        print("The creature is", self.creature)
        print("The final count is", self.count)

if __name__ == '__main__':
    CounterBranchFlow()
