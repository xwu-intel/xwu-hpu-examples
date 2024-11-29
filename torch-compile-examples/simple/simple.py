import torch

def test_graph_simple():
    input_shapes = [(3,6,4), (3,8,4), (3, 10,4)]

    def raw_function(t1,t2):
        tmp1 = torch.mul(t1, t2)
        tmp2 = tmp1.sinc()
        return tmp2.sigmoid()

    compiled_fn = torch.compile(raw_function, backend="hpu_backend")

    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=True)
        t2 = torch.randn(s, requires_grad=True)
        result = raw_function(t1, t2)

        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.cpu(), result, rtol=1e-5)

test_graph_simple()