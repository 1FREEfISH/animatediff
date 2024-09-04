import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

MODEL_TYPES = {
    "eps": "noise",
    "v": "v"
}

class DPMSolverSampler(object):
    def __init__(self, model: torch.nn.Module, device: torch.device = torch.device("cuda"), **kwargs) -> None:
        """
        Initialize the DPMSolverSampler.

        Args:
            model (torch.nn.Module): The model to use for sampling.
            device (torch.device, optional): The device to use. Defaults to torch.device("cuda").
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.model = model
        self.device = device
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name: str, attr: torch.Tensor) -> None:
        """
        Register a buffer in the module.

        Args:
            name (str): The name of the buffer.
            attr (torch.Tensor): The tensor to register as a buffer.
        """
        if isinstance(attr, torch.Tensor):
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               S: int,
               batch_size: int,
               shape: tuple,
               conditioning: dict = None,
               callback: callable = None,
               normals_sequence: list = None,
               img_callback: callable = None,
               quantize_x0: bool = False,
               eta: float = 0.,
               mask: torch.Tensor = None,
               x0: torch.Tensor = None,
               temperature: float = 1.,
               noise_dropout: float = 0.,
               score_corrector: callable = None,
               corrector_kwargs: dict = None,
               verbose: bool = True,
               x_T: torch.Tensor = None,
               log_every_t: int = 100,
               unconditional_guidance_scale: float = 1.,
               unconditional_conditioning: torch.Tensor = None,
               **kwargs) -> tuple:
        """
        Perform sampling using the DPM Solver.

        Args:
            S (int): Number of steps.
            batch_size (int): Batch size for sampling.
            shape (tuple): Shape of the samples (C, H, W).
            conditioning (dict, optional): Conditioning information. Defaults to None.
            callback (callable, optional): Callback function. Defaults to None.
            normals_sequence (list, optional): Sequence of normals. Defaults to None.
            img_callback (callable, optional): Image callback function. Defaults to None.
            quantize_x0 (bool, optional): Flag for quantizing x0. Defaults to False.
            eta (float, optional): Eta parameter. Defaults to 0..
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.
            x0 (torch.Tensor, optional): Initial x0 tensor. Defaults to None.
            temperature (float, optional): Temperature parameter. Defaults to 1..
            noise_dropout (float, optional): Noise dropout parameter. Defaults to 0..
            score_corrector (callable, optional): Score corrector. Defaults to None.
            corrector_kwargs (dict, optional): Keyword arguments for the score corrector. Defaults to None.
            verbose (bool, optional): Verbose flag. Defaults to True.
            x_T (torch.Tensor, optional): Initial x_T tensor. Defaults to None.
            log_every_t (int, optional): Log interval. Defaults to 100.
            unconditional_guidance_scale (float, optional): Guidance scale for unconditional sampling. Defaults to 1..
            unconditional_conditioning (torch.Tensor, optional): Conditioning tensor for unconditional sampling. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: Sampled tensor and additional information.
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                if isinstance(ctmp, torch.Tensor):
                    cbs = ctmp.shape[0]
                    if cbs != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}")
            else:
                if isinstance(conditioning, torch.Tensor):
                    if conditioning.shape[0] != batch_size:
                        print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type=MODEL_TYPES[self.model.parameterization],
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=2,
                              lower_order_final=True)

        return x.to(device), None

