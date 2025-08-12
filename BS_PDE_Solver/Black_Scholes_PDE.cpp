#include<iostream>
#include<cmath>
#include<vector>
#include<algorithm>
#include<fstream>
#include<stdexcept>

// Algorithm for solving a tridiagonal system of linear equations
std::vector<double> tridiagonal_solver(std::vector<double> &a,
                                       std::vector<double> &b,
                                       std::vector<double> &c,
                                       std::vector<double> &d)
{
    // a, b and c constitutes the tri-diagonal band matrix. 
    // a is the lower diagonal, b is the main diagonal and c is the upper diagonal
    int m = b.size();
    if(a.size()!=m-1 || c.size()!=m-1 || d.size()!= m)
    {
        throw std::invalid_argument("Sizes of vectors constituting the tri-diagonal matrix are inconsistent");
    }

    // Converting the tri-diagonal matrix into upper diagonal form
    for(int i=1; i<m; i++)
    {
        double weight = a[i-1] / b[i-1];
        b[i] -= weight * c[i-1];
        d[i] -= weight * d[i-1];
    }
    // caution: the operations done here changes b and d in place. 

    // now substitute back from the reverse
    std::vector<double> v(m);
    v[m-1] = d[m-1]/b[m-1];

    for(int i=m-2; i>=0; i--)
    {
        v[i] = (d[i] - c[i]* v[i+1]) / b[i];
    }
    return v;
}

enum class OptionType {Call, Put};

// Crank–Nicolson Black–Scholes PDE solver for European options
void black_scholes_cn(OptionType type,
                      double S_max,
                      double K,
                      double r,
                      double sigma,
                      double T,
                      int num_t_steps,
                      int num_S_steps,
                      const std::string &out_path)
{
    // Grids
    const double dS = S_max / static_cast<double>(num_S_steps);
    const double dt = T / static_cast<double>(num_t_steps);

    std::vector<double> S_grid(num_S_steps + 1);
    for(int i = 0; i <= num_S_steps; ++i)
    {
        S_grid[i] = i * dS;
    }

    // Initial condition (tau=0): payoff at maturity
    std::vector<double> V(num_S_steps + 1);
    if(type == OptionType::Call)
    {
        for(int i = 0; i <= num_S_steps; i++)
        {
            V[i] = std::max(S_grid[i] - K, 0.0);
        }
    }
    else
    {
        for(int i = 0; i <= num_S_steps; i++)
        {
            V[i] = std::max(K - S_grid[i], 0.0);
        }
    }

    // Build the discretized operator for the interior nodes i = 1 ... num_S_steps-1
    std::vector<double> a_int(num_S_steps+1), b_int(num_S_steps+1), c_int(num_S_steps+1);

    for(int i=1; i<=num_S_steps-1; ++i)
    {
        a_int[i] = 0.25 * (sigma*sigma * i * i - r * i );
        b_int[i] = -0.5 * (sigma*sigma * i * i + r);
        c_int[i] = 0.25 * (sigma*sigma * i * i + r * i);
    }

    // Precompute LHS tridiagonal (I - 0.5*dt*A)
    std::vector<double> lower(num_S_steps, 0.0);       // length num_S_steps (rows 1..num_S_steps)
    std::vector<double> diag_base(num_S_steps+1, 1.0); // length num_S_steps+1 (rows 0..num_S_steps)
    std::vector<double> upper(num_S_steps, 0.0);       // length num_S_steps (rows 0..num_S_steps-1)

    for(int i=1; i<=num_S_steps-1; i++)
    {
        lower[i-1] = - dt * a_int[i];
        diag_base[i] = 1.0 - dt * b_int[i];
        upper[i] = - dt * c_int[i];
    }

    // Imposing boundary conditions on the tri-diagonal matrix
    diag_base[0] = 1.0;
    diag_base[num_S_steps] = 1.0;

    upper[0] = 0.0;      // row 0
    lower[num_S_steps-1] = 0.0;    // row num_S_steps

    // Time-marching in tau from 0 -> T
    std::vector<double> diag(num_S_steps+1);
    std::vector<double> rhs(num_S_steps+1);
    for(int n=0; n<num_t_steps; n++)
    {
        const double tau_n   = n * dt;
        const double tau_np1 = (n+1) * dt;

        // apply Dirichlet BCs to V at time tau_n 
        // this imposition is different from solving the heat equation, 
        // the D. boundary condition is time dependent so it has to be handeled inside the time loop
        if(type == OptionType::Call)
        {
            V.front() = 0.0;
            V.back()  = S_max - K * std::exp(-r * tau_n);
        }
        else
        {
            V.front() = K * std::exp(-r * tau_n);
            V.back()  = 0.0;
        }

        // Build the right side (d vector) : (I + dt * A) V^n
        for(int i=1; i<=num_S_steps-1; ++i)
        {
            rhs[i] = V[i] + dt * (a_int[i] * V[i-1] + b_int[i] * V[i] + c_int[i] * V[i+1]);
        }
        
        // Boundary values in RHS at time tau_{n+1}
        if(type == OptionType::Call)
        {
            rhs[0] = 0.0;
            rhs[num_S_steps] = S_max - K * std::exp(-r * tau_np1);
        }
        else
        {
            rhs[0] = K * std::exp(-r * tau_np1);
            rhs[num_S_steps] = 0.0;
        }

        // copy diagonal (since the tridiadonal solver mutates it)
        diag = diag_base;

        V = tridiagonal_solver(lower, diag, upper, rhs);
    }

    // Write solution at t=0 (tau=T)
    std::ofstream fout(out_path);
    fout << "S V(S)" << '\n';
    for(int i=0; i<=num_S_steps; ++i)
    {
        fout << S_grid[i] << ' ' << V[i] << '\n';
    }
    fout.close();
    std::cout << "Wrote BS solution to " << out_path << std::endl;
}

int main()
{
    // example of an european put option 
    const double K = 100.0; // strike price
    const double r = 0.05; // interest rate
    const double sigma = 0.20; // constant volatility
    const double T = 1.0;  // duration of the contract
    const double S_max = 2.0 * K;   // upper bound on spot set as twice the strike price
    const int    num_S_steps = 400;  // spatial steps
    const int    num_t_steps = 2000; // time steps

    black_scholes_cn(OptionType::Call, S_max, K, r, sigma, T, num_t_steps, num_S_steps, "european_bs_call_solution.txt");
    black_scholes_cn(OptionType::Put, S_max, K, r, sigma, T, num_t_steps, num_S_steps, "european_bs_put_solution.txt");

    return 0;
}
