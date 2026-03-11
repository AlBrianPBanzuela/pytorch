import torch
import torch._library.custom_ops as _custom_ops

# Toggle experiments independently (set to False to disable)
_custom_ops._ENABLE_ALIASING_CHECK = True
_custom_ops._ENABLE_AUTOGRAD_DISPATCH = True

@torch.library.custom_op("foo::bar", mutates_args=())
def bar(a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        e: torch.Tensor,
        f: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        k: torch.Tensor,
        l: torch.Tensor,
        m: torch.Tensor,
        n: torch.Tensor) -> torch.Tensor:
    return b.clone()

@torch.library.custom_op("foo::baz", mutates_args=(["a"]))
def baz(a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        e: torch.Tensor,
        f: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        k: torch.Tensor,
        l: torch.Tensor,
        m: torch.Tensor,
        n: torch.Tensor) -> torch.Tensor:
    return b.clone()

def barbaz(a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        e: torch.Tensor,
        f: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        k: torch.Tensor,
        l: torch.Tensor,
        m: torch.Tensor,
        n: torch.Tensor) -> torch.Tensor:
    return b.clone()

foo_lib = torch.library.Library("foo", "FRAGMENT")

def direct_register_custom_op(
    op_name,
    op_func,
    mutates_args
):
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    foo_lib.define(op_name + schema_str)
    foo_lib.impl(op_name, op_func, "CUDA")

direct_register_custom_op("foo::bar_op", barbaz, mutates_args=())
direct_register_custom_op("foo::baz_op", barbaz, mutates_args=(["a"]))

a = torch.rand([0,128], device="cuda")
b = torch.rand([0,128], device="cuda")
c = torch.rand([0,128], device="cuda")
d = torch.rand([0,128], device="cuda")
e = torch.rand([0,128], device="cuda")
f = torch.rand([0,128], device="cuda")
g = torch.rand([0,128], device="cuda")
h = torch.rand([0,128], device="cuda")
i = torch.rand([0,128], device="cuda")
j = torch.rand([0,128], device="cuda")
k = torch.rand([0,128], device="cuda")
l = torch.rand([0,128], device="cuda")
m = torch.rand([0,128], device="cuda")
n = torch.rand([0,128], device="cuda")

a_grad = torch.rand([0,128], device="cuda", requires_grad=True)
b_grad = torch.rand([0,128], device="cuda", requires_grad=True)
c_grad = torch.rand([0,128], device="cuda", requires_grad=True)
d_grad = torch.rand([0,128], device="cuda", requires_grad=True)
e_grad = torch.rand([0,128], device="cuda", requires_grad=True)
f_grad = torch.rand([0,128], device="cuda", requires_grad=True)
g_grad = torch.rand([0,128], device="cuda", requires_grad=True)
h_grad = torch.rand([0,128], device="cuda", requires_grad=True)
i_grad = torch.rand([0,128], device="cuda", requires_grad=True)
j_grad = torch.rand([0,128], device="cuda", requires_grad=True)
k_grad = torch.rand([0,128], device="cuda", requires_grad=True)
l_grad = torch.rand([0,128], device="cuda", requires_grad=True)
m_grad = torch.rand([0,128], device="cuda", requires_grad=True)
n_grad = torch.rand([0,128], device="cuda", requires_grad=True)


def test():
    from triton.testing import do_bench

    iter = 1000

    def mutate():
        for z in range(iter):
            o = torch.ops.foo.baz(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    def no_mutate():
        for z in range(iter):
            o = torch.ops.foo.bar(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    def no_mutate_grad():
        for z in range(iter):
            o = torch.ops.foo.bar(a_grad, b_grad, c_grad, d_grad, e_grad, f_grad, g_grad, h_grad, i_grad, j_grad, k_grad, l_grad, m_grad, n_grad)

    def mutate_grad():
        for z in range(iter):
            o = torch.ops.foo.baz(a_grad, b_grad, c_grad, d_grad, e_grad, f_grad, g_grad, h_grad, i_grad, j_grad, k_grad, l_grad, m_grad, n_grad)

    def direct_mutate():
        for z in range(iter):
            o = torch.ops.foo.baz_op(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    def direct_no_mutate():
        for z in range(iter):
            o = torch.ops.foo.bar_op(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    def baseline():
        for z in range(iter):
            o = barbaz(a, b, c, d, e, f, g, h, i, j, k, l, m, n)

    def baseline_grad():
        for z in range(iter):
            o = barbaz(a_grad, b_grad, c_grad, d_grad, e_grad, f_grad, g_grad, h_grad, i_grad, j_grad, k_grad, l_grad, m_grad, n_grad)

    with torch.no_grad():
        mutate_time = do_bench(mutate)
        no_mutate_time = do_bench(no_mutate)
    no_mutate_grad_time = do_bench(no_mutate_grad)
    mutate_grad_time = do_bench(mutate_grad)
    direct_mutate_time = do_bench(direct_mutate)
    direct_no_mutate_time = do_bench(direct_no_mutate)
    baseline_time = do_bench(baseline)
    baseline_grad_time = do_bench(baseline_grad)

    print(f"no_mutate = {no_mutate_time}")
    print(f"mutate = {mutate_time}")
    print(f"no_mutate_grad = {no_mutate_grad_time}")
    print(f"mutate_grad = {mutate_grad_time}")
    print(f"direct_no_mutate = {direct_no_mutate_time}")
    print(f"direct_mutate = {direct_mutate_time}")
    print(f"baseline (raw fn call) = {baseline_time}")
    print(f"baseline_grad (raw fn call) = {baseline_grad_time}")

test()
