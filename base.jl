using Jacobi
using LinearAlgebra
using Plots

function element_rhs(u_i, u_left, D, M, h)
    r_i = D' * M * u_i
    r_i[1] += u_left[end]
    r_i[end] -= u_i[end]
    r_i = (2.0 / h) * (M \ r_i)
    return r_i
end

function global_rhs(u, D, M, h)
    N = size(u, 1)
    r = zeros(size(u))
    for i in 1:N
        left_index = i == 1 ? N : i - 1
        u_left = u[left_index, :]
        r[i, :] = element_rhs(u[i, :], u_left, D, M, h)
    end
    return r
end

function to_physical_element(x_l, x_r, xi)
    return (x_r + x_l) / 2.0 + xi * (x_r - x_l) / 2.0
end

function interpolate_global(f, N, nodes, domain)
    u = zeros(N, length(nodes))
    h = (domain[2] - domain[1]) / N
    x_l = domain[1]
    x_r = x_l + h
    for i in 1:N
        for j in eachindex(nodes)
            u[i, j] = f(to_physical_element(x_l, x_r, nodes[j]))
        end
        x_l += h
        x_r += h
    end
    return u
end

function rk3_step(u, dt, rhs)
    u1 = u + dt * rhs(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
    u_next = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2))
    return u_next
end

function main()
    p = 3
    N = 20
    domain = (0.0, 2π)
    T = 1.0
    h = (domain[2] - domain[1]) / N
    dt = h / (2p + 1)
    nodes = zglj(p + 1)
    weights = wglj(nodes)
    M = diagm(weights)
    D = dglj(nodes)
    u0_fun(x) = sin(x)
    u = interpolate_global(u0_fun, N, nodes, domain)
    rhs(u_current) = global_rhs(u_current, D, M, h)
    t = 0.0
    while t < T
        current_dt = min(dt, T - t)
        u = rk3_step(u, current_dt, rhs)
        t += current_dt
    end
    return u
end

function plot_solution(u, nodes, domain)
    N = size(u, 1)
    h = (domain[2] - domain[1]) / N
    xs = Float64[]
    ys = Float64[]
    x_l = domain[1]
    for i in 1:N
        x_r = x_l + h
        for j in eachindex(nodes)
            push!(xs, to_physical_element(x_l, x_r, nodes[j]))
            push!(ys, u[i, j])
        end
        x_l += h
    end
    p = plot(xs, ys, marker=:circle, label="numerisch")
    exact = [sin(x - 1.0) for x in xs]
    plot!(p, xs, exact, label="exakt")
    return p
end

u = main()
p = 3
nodes = zglj(p + 1)
domain = (0.0, 2π)
display(plot_solution(u, nodes, domain))
readline()