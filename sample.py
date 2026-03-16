import torch
from torch._subclasses.fake_tensor import FakeTensorConverter
from torch.fx.experimental.symbolic_shapes import ShapeEnv

shape_env = ShapeEnv()
converter = FakeTensorConverter()

torch._C._create_and_enter_fake_tensor_mode(converter, shape_env)
try:
    # --- Concrete shapes ---
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    c = a + b
    print(f"c.shape = {c.shape}")
    print(f"is_fake = {torch._C._is_fake_tensor(c)}")

    # --- Dynamic shapes ---
    x = torch.randn(5, 8)
    y = torch.randn(5, 8)

    print(f"\nfake_x.shape = {x.shape}")
    print(f"fake_y.shape = {y.shape}")
    print(f"fake_x.shape[0] = {x.shape[0]}")

    s0 = x.shape[0]
    s1 = x.shape[1]
    print(f"s0 + s1 = {s0 + s1}")
    print(f"s0 * 2 = {s0 * 2}")
    print(f"s0 == fake_y.shape[0]: {s0 == y.shape[0]}")
finally:
    torch._C._exit_fake_tensor_mode()
