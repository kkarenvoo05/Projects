using LinearAlgebra
using DataFrames
using CSV
using IterTools
using Graphs
using GraphPlot
using GraphRecipes
using Plots
using SpecialFunctions

# Creates new struct: name for the variable name and r for the number of states
struct Variable
    name::String
    r::Int
end

function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

# Function that extracts the counts of a discrete dataset
function statistics(vars, graph, data)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(graph,i)]) for i in 1:n]
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(data)
        for i in 1:n
            k = o[i]
            parents = inneighbors(graph,i)
            j = 1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
            if k > 0 && k <= r[i] && j > 0 && j <= size(M[i], 1)
                M[i][j, k] += 1.0  # Update count
            else
                continue
            end
        end
    end
    return M
end

# Function that generates the prior
function prior(vars, graph)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(graph,i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

# Function that calculates the Bayesian score for a single variable
function bayesian_score_component(M, α)
    p = sum(loggamma.(α + M))
    p -= sum(loggamma.(α))
    p += sum(loggamma.(sum(α,dims=2)))
    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2)))
    return p
end

# Function that calculates the Bayesian score for all variables in the given graph
function bayesian_score(vars, G, D)
    n = length(vars)
    M = statistics(vars, G, D)
    α = prior(vars, G)
    return sum(bayesian_score_component(M[i], α[i]) for i in 1:n)
end

function check_acyclic(graph)
    return isempty(Graphs.simplecycles(graph)) # Checks if there are no cycles
end

# Function for finding the structure with the largest Bayesian score
function find_optimal_structure(vars, data)
    n = length(vars)
    graph = SimpleDiGraph(n)  # Initialize empty graph with n nodes (variables)

    for i in 1:n
        current_score = bayesian_score(vars, graph, data)  # Bayes score before adding/removing edge
        while true
            best_score = -Inf # The initial score will be the baseline
            best_j = 0
            for j in 1:n  
                if !has_edge(graph, j, i) && j != i  # Ignoring edges with the considered node
                    add_edge!(graph, j, i)

                    if check_acyclic(graph)  # Ensure graph remains acyclic
                        new_score = bayesian_score(vars, graph, data)
                        if new_score > best_score
                            best_score = new_score # Updating current score
                            best_j = j 
                        end
                    end
                    rem_edge!(graph, j, i) # Removes edge to continue checking
                end
            end
            if best_score > current_score  # Permanently add the best edge
                current_score = best_score
                add_edge!(graph, best_j, i) 
            else
                break
            end
        end
    end
    return graph
end

# Function that reads the edges from the optimal structure
function output(graph, filename, vars)
    open(filename, "w") do f
        for edge in edges(graph)
            parent_name = vars[src(edge)].name  # Name of the parent variable
            child_name = vars[dst(edge)].name   # Name of the child variable
            write(f, "$parent_name,$child_name\n")  # Write names instead of numbers
        end
    end
end

# Function that creates the graph graphics
function plot_graph(graph, vars)
    if ne(graph) == 0 || nv(graph) == 0
        println("There exists no edges and/or no vertices.")
        return
    end
    variable_names = [vars[i].name for i in 1:nv(graph)]
    # Plot the graph
    plot = graphplot(graph,
                    nodeshape=:circle,
                    nodesize=0.2,
                    nodecolor=:lightblue,
                    names=variable_names,
                    fontsize=8,
                    linecolor=:black,
                    arrows=true,
                    curves=false,
                    title="Bayesian Network for Small Dataset",
                    legend=false)
    savefig(plot, "small.png")
    println("The graph plot is saved as 'small.png'")
end

# Function that returns a dataset's r_values
function generate_r_val(filename)
     # r_values are the number of states that each variable within each dataset has
    if filename == "small.gph"
        r_values = [3, 3, 3, 3, 3, 2, 3, 2]
    end
    if filename == "medium.gph"
        r_values = [2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8]
    end
    if filename == "large.gph"
        r_values = [3, 3, 4, 2, 2, 4, 3, 2, 2, 2, 3, 2, 3, 3, 3, 4, 2, 2, 3, 2, 3, 2, 4, 4, 3, 4, 4, 3, 2, 2, 4, 3, 3, 4, 4, 3, 2, 2, 2, 4, 4, 3, 2, 2, 2, 2, 4, 2]
    end
    return r_values
end

function main()
    filename = "small.gph"
    data = CSV.read("/Users/karenvo/Downloads/small.csv", DataFrame)

    r_values = generate_r_val(filename)
    
    vars = [Variable(name, r) for (name, r) in zip(names(data), r_values)]
    dataArray = Matrix{Int}(data)

    optimal = find_optimal_structure(vars, dataArray)  # Use the function to find optimal structure
    output(optimal, filename, vars)
    
    score = bayesian_score(vars, optimal, dataArray)
    println("Bayesian Score: $score")

    print("Success\n")
    open(filename, "r") do f
        println(read(f, String))
    end
    plot_graph(optimal, vars)    
end

main()
