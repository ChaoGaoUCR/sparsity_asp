import types
import torch
from .sparse_masklib import create_mask
from .permutation_lib import Permutation

torchvision_imported=True
try:
    import torchvision
except ImportError:
    print("[ASP][Warning] torchvision cannot be imported.")
    torchvision_imported=False

import json
import os
import string
import time
def count_matching_and_total_elements(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape.")
    matches = tensor1 == tensor2
    num_matches = matches.sum().item()  # .item() to convert to Python scalar
    total_elements = tensor1.numel()  # .numel() returns the total number of elements in the tensor
    return num_matches, total_elements
def eligible_modules(model, whitelist_layer_types, allowed_layer_names, disallowed_layer_names):
    eligible_modules_list = []
    for name, mod in model.named_modules():
        if isinstance(mod, whitelist_layer_types) and name not in disallowed_layer_names:
            if allowed_layer_names is not None and name not in allowed_layer_names:
                continue
            eligible_modules_list.append((name, mod))
    # breakpoint()
    return eligible_modules_list


class ASP:
    __model = []
    __verbosity = 0
    __optimizer = []
    __sparse_parameters = []
    __calculate_mask = []
    __allow_permutation = False
    __all_parameters = []
    __save_permutation_graph = False
    __permutation_output_dir = ''
    # __model1 = None
    __model_index = 0
    
    @classmethod
    def clean_ASP(cls):
        cls.__model = []
        cls.__verbosity = 0
        cls.__optimizer = []
        cls.__sparse_parameters = []
        cls.__calculate_mask = []
        cls. __allow_permutation = False
        cls.__all_parameters = []
        cls.__save_permutation_graph = False
        cls.__permutation_output_dir = ''
        # cls.__model1 = None
        cls.__model_index = 0

    @classmethod
    def init_model_for_pruning(cls, model, mask_calculator="m4n2_1d",
             verbosity=3,
             whitelist=[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.MultiheadAttention], 
             allowed_layer_names=None, disallowed_layer_names=[],
             allow_recompute_mask=False, custom_layer_dict={},
             allow_permutation=False):
        # assert (cls.__model is None), "ASP has been initialized already."
        cls.__model.append(model)
        cls.__verbosity = verbosity
        cls.__model_index = len(cls.__model) - 1
        cls.__allow_permutation = allow_permutation

        if isinstance(mask_calculator, str):
            def create_mask_from_pattern(param,grad):
                return create_mask(param, mask_calculator,tensor_grad=grad).bool()
            cls.__calculate_mask.append(create_mask_from_pattern)
        # else:
        #     cls.__calculate_mask = mask_calculator #user defined function

        if torchvision_imported:
            print("[ASP] torchvision is imported, can work with the MaskRCNN/KeypointRCNN from torchvision.")
            torchvision_version = str(torchvision.__version__)
            torchvision_version_major = int(torchvision_version.split('.')[0])
            torchvision_version_minor = int(torchvision_version.split('.')[1])
            if torchvision_version_major == 0 and torchvision_version_minor < 12:
                sparse_parameter_list = {torch.nn.Linear: ['weight'], torch.nn.Conv1d: ['weight'], torch.nn.Conv2d: ['weight'], torch.nn.Conv3d: ['weight'], torch.nn.modules.linear.NonDynamicallyQuantizableLinear: ['weight'], torch.nn.MultiheadAttention: ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight'], torchvision.ops.misc.Conv2d: ['weight']}
            else:    # Torchvision remove APIs that were deprecated before 0.8 (#5386) in 0.12.0, torchvision.ops.misc.Conv2d is removed
                sparse_parameter_list = {torch.nn.Linear: ['weight'], torch.nn.Conv1d: ['weight'], torch.nn.Conv2d: ['weight'], torch.nn.Conv3d: ['weight'], torch.nn.modules.linear.NonDynamicallyQuantizableLinear: ['weight'], torch.nn.MultiheadAttention: ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']}
        else:
            sparse_parameter_list = {torch.nn.Linear: ['weight'], torch.nn.Conv1d: ['weight'], torch.nn.Conv2d: ['weight'], torch.nn.Conv3d: ['weight'], torch.nn.modules.linear.NonDynamicallyQuantizableLinear: ['weight'], torch.nn.MultiheadAttention: ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']}
        if custom_layer_dict: # Update default list to include user supplied custom (layer type : parameter tensor), make sure this tensor type is something ASP knows how to prune
            sparse_parameter_list.update(custom_layer_dict)
            whitelist += list(custom_layer_dict.keys())
        sparse_parameters = []
        for module_type in whitelist:
            assert (module_type in sparse_parameter_list), "Module %s :: Don't know how to sparsify module." % module.dtype()
        def add_sparse_attributes(module_name, module, sparse_parameters_):
            sparse_parameters = sparse_parameter_list[type(module)]
            for p_name, p in module.named_parameters():
                if p_name in sparse_parameters and p.requires_grad:
                    # check for NVIDIA's TC compatibility: we check along the horizontal direction
                    if p.dtype == torch.float32 and ((p.size()[0] % 8) != 0 or (p.size()[1] % 16) != 0): #User defines FP32 and APEX internally uses FP16 math
                        print("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                        continue
                    if p.dtype == torch.float16 and ((p.size()[0] % 8) != 0 or (p.size()[1] % 16) != 0): #For Conv2d dim= K x CRS; we prune along C
                        print("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                        continue
                    
                    if cls.__verbosity >= 3:
                        print("[ASP] Sparsifying %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                    
                    mask = torch.ones_like(p).bool()
                    buffname = p_name.split(".")[-1] # buffer names cannot contain "."
                    module.register_buffer('__%s_mma_mask' % buffname, mask)
                    # if allow_recompute_mask:
                    #     pruned = torch.zeros_like(p).cpu()
                    #     module.register_buffer('__%s_mma_pruned_p' % buffname, pruned)
                    # else:
                    #     pruned = None
                    # cls.__sparse_parameters.append((module_name, module, p_name, p, mask, pruned))
                    sparse_parameters_.append((module_name, module, p_name, p, mask, None))
                # else:
                #     if cls.__verbosity >= 3:
                #         print("[ASP] Not sparsifying %s::%s of size=%s and type=%s" % (module_name, p_name, str(p.size()), str(p.dtype)))
            # return module_name, module, p_name, p, mask, None
        for name, sparse_module in eligible_modules(model, tuple(whitelist), allowed_layer_names, disallowed_layer_names):
            add_sparse_attributes(name, sparse_module, sparse_parameters)
        cls.__sparse_parameters.append(sparse_parameters)


    @classmethod
    def init_optimizer_for_pruning(cls, optimizer):
        """Call this method to monkey patch optimizer step function so that masks can be applied to
        gradients and weights during training.
        You must call init_model_for_pruning(...) before calling init_optimizer_for_pruning(...)
        """
        cls.__optimizer.append(optimizer)
        cls.__optimizer[-1].__step = optimizer.step
        sparse_parameters = cls.__sparse_parameters[cls.__model_index]

        def __step(opt_self, *args, **kwargs):
            # prune gradients before step method
            with torch.no_grad():
                for module_name, module, p_name, p, mask, pruned in sparse_parameters:
                    if p.grad is not None: #thx pjudd
                        p.grad.mul_(mask)
            # call original optimizer step method
            rval = opt_self.__step(*args, **kwargs)
            # prune parameters after step method
            with torch.no_grad():
                for module_name, module, p_name, p, mask, pruned in sparse_parameters:
                    p.mul_(mask)
            return rval
        cls.__optimizer[cls.__model_index].step = types.MethodType(__step, cls.__optimizer)

    @classmethod
    def compute_sparse_masks(cls):
        with torch.no_grad():
            sparse_parameters = cls.__sparse_parameters[cls.__model_index]
            # breakpoint()
            for module_name, module, p_name, p, mask, pruned in sparse_parameters:
                if mask.sum() < mask.numel(): # when recalculating masks
                    # restore dense parameter if allow_recompute_mask is enabled
                    assert (pruned is not None), "Unable to restore dense parameter because allow_recompute_mask == False"
                    p.add_(pruned.cuda())
                calculate_mask = cls.__calculate_mask[cls.__model_index]
                mask.set_(calculate_mask(p, module.weight.grad))

                if pruned is not None: # stow away pruned weights to cpu
                    pruned.set_((p * (~mask)).cpu())

                p.mul_(mask) # in-place multiplication, so pruned weights are 0-values, hence checkpoint will have 0s for pruned weights
                if cls.__verbosity >= 2:
                    print("[ASP] Enabled %.2f%% sparsity for %s::%s of size=%s and type=%s with magnitude %s" % (100.0-100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype), torch.sum(torch.abs(p))))

    @classmethod
    def is_sparsity_enabled(cls):
        """Call this method to determine if sparsity is enabled in the model.
        The typical use case is right after checkpoint has been loaded.
        """
        total,sp100,sp50 = 0,0,0
        sparse_parameters = cls.__sparse_parameters[cls.__model_index]
        for module_name, module, p_name, p, mask, pruned in sparse_parameters:
            total += 1
            mask_sum = mask.sum()
            mask_numel = mask.numel()
            if mask_sum == mask_numel:
                sp100 += 1
            elif mask_sum*2 == mask_numel:
                sp50 += 1

        assert (total == sp100 or total == sp50), "Inconsistent model sparsity"
        if total == sp100:
            return False
        elif total == sp50:
            return True
    
    @classmethod
    def prune_trained_model(cls, model, optimizer,mask_calculator="m8n7"):
        cls.init_model_for_pruning(model, mask_calculator=mask_calculator+'_1d', verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention], allow_recompute_mask=False)
        cls.init_optimizer_for_pruning(optimizer)
        cls.compute_sparse_masks()

    @classmethod
    def cound_two_model_mask(cls, index_1, index_2):
        diff_list = []
        sparse_1 = cls.__sparse_parameters[index_1]
        sparse_2 = cls.__sparse_parameters[index_2]
        assert len(sparse_1) == len(sparse_2), "Wrong Masks comparison"
        for i in range(len(sparse_1)):
            mask1 = sparse_1[i][4]
            mask2 = sparse_2[i][4]
            match_mask, total = count_matching_and_total_elements(mask1, mask2)
            diff_list.append([match_mask, total])
        return diff_list
            
        