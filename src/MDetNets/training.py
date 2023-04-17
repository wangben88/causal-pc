from src.MDetNets.MDNet import LazyDenseMDNet
import jax.numpy as jnp
from jax.example_libraries.optimizers import sgd, adam
import jax
import functools
from jax import jit

def unnorm_log_like(net, inputs):
    outputs = net.forward_log(inputs)
    return jnp.squeeze(jnp.sum(outputs, axis=0))  # [0, 0]

def loss(net, inputs):

    outputs = net.forward_log(inputs)
    unnorm_log_like = jnp.squeeze(jnp.sum(outputs, axis=0))#[0, 0]

    norm_log_like = unnorm_log_like - len(outputs) * jnp.squeeze(net.forward_log({}))#[0, 0, 0]

    return -norm_log_like

def train(net, inputs_batched, num_steps):
    init_fun, update_fun, get_params = adam(step_size=0.02)

    opt_state = init_fun(net)

    def step(step, opt_state, inputs_batched):
        tot_value = 0
        for input_batch in inputs_batched:
            value, grads = jax.value_and_grad(loss, argnums=0)(get_params(opt_state), input_batch)
            tot_value += value
            opt_state = update_fun(step, grads, opt_state)

        return tot_value, opt_state

    for i in range(num_steps):
        value, opt_state = step(i, opt_state, inputs_batched)
        if i == 0:
            print("step 0 done")
        if i == 1:
            print("step 1 done")
        if i % 10 == 0:
            print("Step ", i, ": ", value)

    return get_params(opt_state)

def train_em(net, inputs_batched, num_steps):

    def step():
        grads_list = []
        tot_value = 0
        for input_batch in inputs_batched:
            value, grads = jax.value_and_grad(unnorm_log_like, argnums=0)(net, input_batch)
            grads_list.append(grads)
            tot_value += value

        for region in net.regions:
            if not net.leaves[region]:
                net.region_weights[region].update_em([grads.region_weights[region] for grads in grads_list])
            else:
                pass

        return tot_value

    for i in range(num_steps):
        value = step()
        if i % 1 == 0:
            print("Step ", i, ": ", value)

    return net

def train_det(net, inputs_batched):
    total_product_counts = {region: None for region in net.regions if not net.leaves[region]}
    tot_value = 0
    for input_batch in inputs_batched:
        outputs, product_counts = net.forward_log(input_batch, return_counts=True)
        _, grads = jax.value_and_grad(unnorm_log_like, argnums=0)(net, input_batch)
        for region in total_product_counts:
            if total_product_counts[region] is None:
                total_product_counts[region] = product_counts[region]
            else:
                total_product_counts[region] = total_product_counts[region] + product_counts[region]
        tot_value += jnp.squeeze(jnp.sum(outputs, axis=0))


    net.update_em_det(total_product_counts)

    new_tot_value = 0
    for input_batch in inputs_batched:
        value = unnorm_log_like(net, input_batch)
        new_tot_value += value






