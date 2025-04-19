import torch

class Wang2020(torch.nn.Module):
    def __init__(self, number_of_inputs, number_of_blocks, block_width, number_of_outputs=1):
        super().__init__()
        self.linear1 = torch.nn.Linear(number_of_inputs, block_width)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(number_of_inputs, block_width)
        self.tanh2 = torch.nn.Tanh()
        self.blocks = torch.nn.ModuleDict()
        self.blocks["block1"] = torch.nn.Sequential(
            torch.nn.Linear(number_of_inputs, block_width),
            torch.nn.Tanh()
        )
        for block_ndx in range(2, number_of_blocks + 1):
            self.blocks[f"block{block_ndx}"] = torch.nn.Sequential(
                torch.nn.Linear(block_width, block_width),
                torch.nn.Tanh()
            )
        self.linear3 = torch.nn.Linear(block_width, number_of_outputs)

    def forward(self, input_tsr):  # (B, N_i)
        act1 = self.linear1(input_tsr)  # (B, W)
        U = self.tanh1(act1)  # (B, W)
        act2 = self.linear2(input_tsr)  # (B, W)
        V = self.tanh2(act2)  # (B, W)
        act = input_tsr
        for block_ndx in range(1, len(self.blocks) + 1):
            Z = self.blocks[f"block{block_ndx}"](act)  # (B, W)
            act = (1.0 - Z) * U + Z * V  # (B, W)
        output_tsr = self.linear3(act)  # (B, N_o)
        return output_tsr

class ResidualBlock(torch.nn.Module):
    def __init__(self, number_of_inputs, number_of_outputs):
        super().__init__()
        self.linear1 = torch.nn.Linear(number_of_inputs, number_of_outputs)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(number_of_outputs, number_of_outputs)
        self.tanh2 = torch.nn.Tanh()
        self.passthrough = torch.nn.Identity()
        if number_of_inputs != number_of_outputs:
            self.passthrough = torch.nn.Sequential(
                torch.nn.Linear(number_of_inputs, number_of_outputs)
            )

    def forward(self, input_tsr):  # input_tsr.shape = (B, Ni)
        act1 = self.linear1(input_tsr)  # (B, No)
        act2 = self.tanh1(act1)  # (B, No)
        act3 = self.linear2(act2)  # (B, No)
        act4 = self.tanh2(act3)  # (B, No)
        act5 = act4 + self.passthrough(input_tsr)  # (B, No)
        return act5

class ResidualNet(torch.nn.Module):
    def __init__(self, number_of_inputs, number_of_blocks, block_width, number_of_outputs):
        super().__init__()
        self.blocks_dict = torch.nn.ModuleDict()
        self.blocks_dict['block1'] = ResidualBlock(number_of_inputs, block_width)
        for block_ndx in range(2, number_of_blocks + 1):
            self.blocks_dict[f'block{block_ndx}'] = ResidualBlock(block_width, block_width)
        self.linear1 = torch.nn.Linear(block_width, number_of_outputs)

    def forward(self, input_tsr):  # input_tsr.shape = (B, Ni)
        act = self.blocks_dict['block1'](input_tsr)  # (B, W)
        for block_ndx in range(2, len(self.blocks_dict) + 1):
            act = self.blocks_dict[f'block{block_ndx}'](act)  # (B, W)
        output_tsr = self.linear1(act)  # (B, No)
        return output_tsr

class MLP(torch.nn.Module):
    def __init__(self, number_of_inputs, layer_widths, number_of_outputs):
        super().__init__()
        self.blocks = torch.nn.ModuleDict()
        self.blocks['layer1'] = torch.nn.Sequential(
            torch.nn.Linear(number_of_inputs, layer_widths[0]),
            torch.nn.Tanh()
        )
        for layer_ndx in range(2, len(layer_widths) + 1):
            self.blocks[f'layer{layer_ndx}'] = torch.nn.Sequential(
                torch.nn.Linear(layer_widths[layer_ndx - 2], layer_widths[layer_ndx - 1]),
                torch.nn.Tanh()
            )
        self.blocks[f'layer{len(layer_widths) + 1}'] = torch.nn.Linear(layer_widths[-1], number_of_outputs)

    def forward(self, input_tsr):  # input_tsr.shape = (B, Ni)
        act = input_tsr
        for layer_ndx in range(1, len(self.blocks) + 1):
            act = self.blocks[f'layer{layer_ndx}'](act)
        return act