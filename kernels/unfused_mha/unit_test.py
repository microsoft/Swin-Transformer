import unittest

import torch
import softmax_cuda
import torch.nn.functional as F
import time


default_dtype = torch.float16

class MySoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, relative_pos_bias, attn_mask, batch_size, window_num, num_head, window_len):
        if attn_mask is not None:
            softmax_cuda.softmax_fwd(input, relative_pos_bias, attn_mask, batch_size, window_num, num_head, window_len)
        else:
            softmax_cuda.softmax_nomask_fwd(input, relative_pos_bias, batch_size, window_num, num_head, window_len)
        
        ctx.save_for_backward(input)
        # save for backward for int/float values
        ctx.batch_size = batch_size
        ctx.window_num = window_num
        ctx.num_head = num_head
        ctx.window_len = window_len
        return input

    @staticmethod
    def backward(ctx, grad_out):
        softmax_result = ctx.saved_tensors[0]
        #_tmp, _sm1, _sm2, delta = cal_grad(grad_out, output)
        batch_size = ctx.batch_size
        window_num = ctx.window_num
        num_head = ctx.num_head
        window_len = ctx.window_len

        softmax_cuda.softmax_bwd(
            grad_out.contiguous(), softmax_result, batch_size, window_num, num_head, window_len)

        return grad_out, torch.sum(grad_out, dim=0), None, None, None, None, None


def copy_tensor(input, pos, mask):
    input1 = input.clone().detach().requires_grad_(True).cuda()
    pos1 = pos.clone().detach().requires_grad_(True).cuda()
    mask1 = mask.clone().detach().requires_grad_(False).cuda()
    return input1, pos1, mask1


def copy_one_tensor(input, requires_grad=False):
    input1 = input.clone().detach().requires_grad_(requires_grad).cuda()
    return input


def generate_tensors(batch_size, window_num, num_head, window_len, use_pos=True, use_mask=True, dtype=default_dtype):
    # for dropout
    #random_tensor = torch.rand((window_num, num_head, window_len, window_len), dtype=torch.float32).cuda()

    # for input data
    input = torch.randn((batch_size*window_num, num_head, window_len, window_len), dtype=dtype, requires_grad=True).cuda()

    # for loss tensor
    d_loss_tensor = torch.randn((batch_size*window_num, num_head, window_len, window_len), dtype=dtype).cuda()

    # save softmax result for backward
    softmax_result = torch.zeros(batch_size*window_num, num_head, window_len, window_len, dtype=dtype).cuda()

    # relative position bias and mask
    if not use_pos:
        pos = torch.zeros(num_head, window_len, window_len, dtype=dtype).cuda()
    else:
        pos = torch.randn(num_head, window_len, window_len, dtype=dtype).cuda()
    
    if not use_mask:
        mask = torch.zeros(window_num, window_len, window_len, dtype=dtype).cuda()
    else:
        mask = torch.randn(window_num, window_len, window_len, dtype=dtype).cuda()

    return input, pos, mask, softmax_result, d_loss_tensor


def print_diff(expected, output, info):
    delta = torch.abs(expected - output)
    percent = torch.div(delta, torch.abs(expected) + 1e-7)

    max_val = torch.max(delta)
    max_percent = torch.max(percent)

    dim = len(expected)
    
    print('INFO:\t{}\n'
        '\tDelta mean:\t{:}\t{:%}\n'
        '\tDelta max:\t{:}\t{:%}'.format(
        info, 
        torch.mean(delta).item(), torch.mean(percent).item() * 100, 
        torch.max(delta).item(), torch.max(percent).item() * 100))

    # max val, check max percentage
    coords = (max_val == delta).nonzero(as_tuple=True)
    n = len(coords[0])
    max_percentage_of_max_val = 0
    for i in range(n):
        if 'relative' not in info:
            val1 = expected[coords[0][i]][coords[1][i]][coords[2][i]][coords[3][i]]
            val2 = output[coords[0][i]][coords[1][i]][coords[2][i]][coords[3][i]]
        else:
            val1 = expected[coords[0][i]][coords[1][i]][coords[2][i]]
            val2 = output[coords[0][i]][coords[1][i]][coords[2][i]]
        max_percentage_of_max_val = max(max_percentage_of_max_val, torch.abs((val1 - val2) / (val1 + 1e-9))) 
    
    # max percentage, check max val
    coords = (max_percent == percent).nonzero(as_tuple=True)
    n = len(coords[0])
    max_val_of_max_percentage = 0
    for i in range(n):
        if 'relative' not in info:
            val1 = expected[coords[0][i]][coords[1][i]][coords[2][i]][coords[3][i]]
            val2 = output[coords[0][i]][coords[1][i]][coords[2][i]][coords[3][i]]
        else:
            val1 = expected[coords[0][i]][coords[1][i]][coords[2][i]]
            val2 = output[coords[0][i]][coords[1][i]][coords[2][i]]
        max_val_of_max_percentage = max(max_val_of_max_percentage, torch.abs(val1 - val2))
    
    print('\t\tmax_percentage_of_max_val:\t{:%}\n'
        '\t\tmax_val_of_max_percentage:\t{}'.format(
            max_percentage_of_max_val * 100, max_val_of_max_percentage
        ))



class Test(unittest.TestCase):
    def setUp(self):
        self.batch_size = 192
        self.window_num = 64
        self.num_head = 6
        self.window_len = 49
        self.dropout_rate = 0.5
    
    def test_forward_no_mask(self, use_pos=True, use_mask=False, dtype=torch.float32):
        input, pos, mask, softmax_result, d_loss_tensor = generate_tensors(self.batch_size, self.window_num, self.num_head, self.window_len, use_pos=use_pos, use_mask=use_mask, dtype=dtype)
        input1, pos1, mask1 = copy_tensor(input, pos, mask)
        input2, pos2, mask2 = copy_tensor(input, pos, mask)

        with torch.no_grad():
            attn = input1 + pos1.unsqueeze(0)
            expected = F.softmax(attn, dim=-1)

        mask2 = None
        with torch.no_grad():
            output = MySoftmax.apply(input2, pos2, mask2, self.batch_size, self.window_num, self.num_head, self.window_len)
        
        if dtype == torch.float32:
            self.assertTrue(torch.allclose(expected, output, rtol=1e-05, atol=1e-08))
        else:
            self.assertTrue(torch.mean(expected - output) < 1.22e-4)
        #print(torch.allclose(expected, output, rtol=1e-05, atol=1e-08))
        #info = 'Forward diff, no mask'
        #print_diff(expected, output, info)

    def test_forward_with_mask(self, use_pos=True, use_mask=True, dtype=torch.float32):
        input, pos, mask, softmax_result, d_loss_tensor = generate_tensors(self.batch_size, self.window_num, self.num_head, self.window_len, use_pos=use_pos, use_mask=use_mask, dtype=dtype)
        input1, pos1, mask1 = copy_tensor(input, pos, mask)
        input2, pos2, mask2 = copy_tensor(input, pos, mask)

        with torch.no_grad():
            attn = input1 + pos1.unsqueeze(0)
            nW = mask.shape[0]
            attn = attn.view(self.batch_size, nW, self.num_head, self.window_len, self.window_len) + mask1.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_head, self.window_len, self.window_len)
            expected = F.softmax(attn, dim=-1)

        with torch.no_grad():
            output = MySoftmax.apply(input2, pos2, mask2, self.batch_size, self.window_num, self.num_head, self.window_len)
        
        if dtype == torch.float32:
            self.assertTrue(torch.allclose(expected, output, rtol=1e-05, atol=1e-08))
        else:
            self.assertTrue(torch.mean(expected - output) < 1.22e-4)
        #print(torch.allclose(expected, output, rtol=1e-05, atol=1e-08))
        #info = 'Forward diff, complete version'
        #print_diff(expected, output, info)

    def test_mask_backward(self, use_pos=True, use_mask=True, dtype=torch.float32):
        input, pos, mask, softmax_result, d_loss_tensor = generate_tensors(self.batch_size, self.window_num, self.num_head, self.window_len, use_pos=True, use_mask=True, dtype=dtype)
        input1, pos1, mask1 = copy_tensor(input, pos, mask)
        input2, pos2, mask2 = copy_tensor(input, pos, mask)

        # SwinTransformer official
        attn = input1 + pos1.unsqueeze(0)
        nW = mask.shape[0]
        attn = attn.view(self.batch_size, nW, self.num_head, self.window_len, self.window_len) + mask1.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_head, self.window_len, self.window_len)
        expected = F.softmax(attn, dim=-1)
        expected.backward(d_loss_tensor)

        # my op
        output = MySoftmax.apply(input2, pos2, mask2, self.batch_size, self.window_num, self.num_head, self.window_len)
        output.backward(d_loss_tensor)
        
        self.assertTrue(torch.mean(input1.grad - input2.grad) < 1.22e-4)
        #print(torch.allclose(expected, output, rtol=1e-05, atol=1e-08))
        # check grad
        #info = 'Backward diff, input gradient'
        #print_diff(input1.grad, input2.grad, info)
        #info = 'Backward diff, relative position gradient'
        #print_diff(pos1.grad, pos2.grad, info)

    def test_nomask_backward(self, use_pos=True, use_mask=True, dtype=torch.float32):
        input, pos, mask, softmax_result, d_loss_tensor = generate_tensors(self.batch_size, self.window_num, self.num_head, self.window_len, use_pos=True, use_mask=False, dtype=dtype)
        input1, pos1, mask1 = copy_tensor(input, pos, mask)
        input2, pos2, mask2 = copy_tensor(input, pos, mask)

        # SwinTransformer official
        attn = input1 + pos1.unsqueeze(0)
        nW = mask.shape[0]
        attn = attn.view(self.batch_size, nW, self.num_head, self.window_len, self.window_len) + mask1.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_head, self.window_len, self.window_len)
        attn0 = F.softmax(attn, dim=-1)
        expected = F.dropout(attn0, self.dropout_rate)
        expected.backward(d_loss_tensor)

        # save for my backward
        th_softmax_result = attn0.detach().clone()

        # my op
        output = MySoftmax.apply(input2, pos2, mask2, self.batch_size, self.window_num, self.num_head, self.window_len)
        output.backward(d_loss_tensor)
        
        self.assertTrue(torch.mean(input1.grad - input2.grad) < 1e-4)
        '''
        print(torch.allclose(expected, output, rtol=1e-05, atol=1e-08))
        # check grad
        info = 'Backward diff, input gradient'
        print_diff(input1.grad, input2.grad, info)
        info = 'Backward diff, relative position gradient'
        print_diff(pos1.grad, pos2.grad, info)
        '''

    def test_forward_backward_speed(self, use_pos=True, use_mask=True, dtype=torch.float32, times=1000):
        input, pos, mask, softmax_result, d_loss_tensor = generate_tensors(self.batch_size, self.window_num, self.num_head, self.window_len, use_pos=True, use_mask=True, dtype=dtype)
        input1, pos1, mask1 = copy_tensor(input, pos, mask)
        input2, pos2, mask2 = copy_tensor(input, pos, mask)

        # SwinTransformer official
        def run_pyt(t=1000):
            for _ in range(t):
                attn = input1 + pos1.unsqueeze(0)
                nW = mask.shape[0]
                attn = attn.view(self.batch_size, nW, self.num_head, self.window_len, self.window_len) + mask1.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_head, self.window_len, self.window_len)
                expected = F.softmax(attn, dim=-1)
                expected.backward(d_loss_tensor)

        # my op
        def run_myop(t=1000):
            for _ in range(t):
                output = MySoftmax.apply(input2, pos2, mask2, self.batch_size, self.window_num, self.num_head, self.window_len)
                output.backward(d_loss_tensor)
        
        torch.cuda.synchronize()
        t1 = time.time()
        run_pyt(t=times)
        torch.cuda.synchronize()
        t2 = time.time()
        run_myop(t=times)
        torch.cuda.synchronize()
        t3 = time.time()
        self.assertTrue((t3 - t2) < (t2 - t1))

        print('Run {} times'.format(times))
        print('pyt time cost: {}'.format(t2 - t1))
        print('myop time cost: {}'.format(t3 - t2))


    def test_forward_no_mask_fp16(self):
        self.test_forward_no_mask(dtype=torch.float16)
    
    def test_forward_with_mask_fp16(self):
        self.test_forward_with_mask(dtype=torch.float16)

    def test_mask_backward_fp16(self):
        self.test_mask_backward(dtype=torch.float16)

    def test_nomask_backward_fp16(self):
        self.test_nomask_backward(dtype=torch.float16)

    def test_forward_backward_speed_fp16(self):
        self.test_forward_backward_speed(dtype=torch.float16)



if __name__ == '__main__':
    torch.manual_seed(0)
    unittest.main(verbosity=2)