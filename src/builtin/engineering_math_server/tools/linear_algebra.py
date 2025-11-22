"""
Linear Algebra & Matrix Mathematics Tools for Engineering Math Server.

This module provides 6 core linear algebra tools for matrix operations,
decompositions, and solving linear systems.
"""

import logging
import numpy as np
from scipy import linalg
from typing import Any, Dict, List, Union

from mcp.types import Tool, TextContent, CallToolResult

logger = logging.getLogger("engineering-math-server")


# ============================================================================
# Tool 1: Matrix Operations
# ============================================================================

def matrix_operations(
    operation: str,
    matrix_a: List[List[float]],
    matrix_b: List[List[float]] = None
) -> Dict[str, Any]:
    """
    Perform basic matrix operations: add, subtract, multiply, transpose, trace, determinant.
    
    Args:
        operation: Operation to perform ('add', 'subtract', 'multiply', 'transpose', 'trace', 'determinant')
        matrix_a: First matrix (2D list)
        matrix_b: Second matrix for binary operations (optional)
        
    Returns:
        Dictionary with operation result
    """
    A = np.array(matrix_a, dtype=float)
    
    if operation == "transpose":
        result = np.transpose(A).tolist()
        return {"result": result, "shape": list(np.transpose(A).shape)}
    
    elif operation == "trace":
        if A.shape[0] != A.shape[1]:
            raise ValueError("Trace requires a square matrix")
        result = float(np.trace(A))
        return {"trace": result}
    
    elif operation == "determinant":
        if A.shape[0] != A.shape[1]:
            raise ValueError("Determinant requires a square matrix")
        det = float(np.linalg.det(A))
        return {"determinant": det, "is_singular": abs(det) < 1e-10}
    
    # Binary operations
    if matrix_b is None:
        raise ValueError(f"Operation '{operation}' requires matrix_b")
    
    B = np.array(matrix_b, dtype=float)
    
    if operation == "add":
        if A.shape != B.shape:
            raise ValueError(f"Matrix shapes must match for addition: {A.shape} vs {B.shape}")
        result = (A + B).tolist()
        return {"result": result, "shape": list((A + B).shape)}
    
    elif operation == "subtract":
        if A.shape != B.shape:
            raise ValueError(f"Matrix shapes must match for subtraction: {A.shape} vs {B.shape}")
        result = (A - B).tolist()
        return {"result": result, "shape": list((A - B).shape)}
    
    elif operation == "multiply":
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Invalid shapes for multiplication: {A.shape} @ {B.shape}")
        result = (A @ B).tolist()
        return {"result": result, "shape": list((A @ B).shape)}
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


# ============================================================================
# Tool 2: Matrix Inverse
# ============================================================================

def matrix_inverse(
    matrix: List[List[float]],
    method: str = "lu",
    regularization: float = 0.0
) -> Dict[str, Any]:
    """
    Compute matrix inverse using various methods.
    
    Args:
        matrix: Input square matrix (2D list)
        method: Inversion method ('lu', 'cholesky', 'svd')
        regularization: Ridge regularization parameter for ill-conditioned matrices
        
    Returns:
        Dictionary with inverse matrix and conditioning information
    """
    A = np.array(matrix, dtype=float)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")
    
    # Add regularization if specified
    if regularization > 0:
        A = A + regularization * np.eye(A.shape[0])
    
    # Compute condition number
    cond_number = float(np.linalg.cond(A))
    is_ill_conditioned = cond_number > 1e10
    
    # Check for singularity
    det = np.linalg.det(A)
    if abs(det) < 1e-12 and regularization == 0:
        return {
            "error": "Matrix is singular (determinant ≈ 0)",
            "determinant": float(det),
            "condition_number": cond_number,
            "suggestion": "Consider adding regularization parameter"
        }
    
    try:
        if method == "lu":
            # LU decomposition based inverse
            inv_matrix = linalg.inv(A)
        
        elif method == "cholesky":
            # Cholesky decomposition (for positive definite matrices)
            try:
                L = linalg.cholesky(A, lower=True)
                inv_matrix = linalg.cho_solve((L, True), np.eye(A.shape[0]))
            except linalg.LinAlgError:
                raise ValueError("Matrix is not positive definite, Cholesky decomposition failed")
        
        elif method == "svd":
            # SVD-based pseudo-inverse (more stable for ill-conditioned matrices)
            inv_matrix = np.linalg.pinv(A)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lu', 'cholesky', or 'svd'")
        
        return {
            "inverse": inv_matrix.tolist(),
            "condition_number": cond_number,
            "is_ill_conditioned": is_ill_conditioned,
            "determinant": float(det),
            "method": method
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "condition_number": cond_number,
            "suggestion": "Try 'svd' method or add regularization"
        }


# ============================================================================
# Tool 3: Matrix Decomposition
# ============================================================================

def matrix_decomposition(
    matrix: List[List[float]],
    decomposition_type: str = "lu"
) -> Dict[str, Any]:
    """
    Perform matrix decomposition (LU, QR, Cholesky, SVD, Eigenvalue).
    
    Args:
        matrix: Input matrix (2D list)
        decomposition_type: Type of decomposition ('lu', 'qr', 'cholesky', 'svd', 'eigen')
        
    Returns:
        Dictionary with decomposition results
    """
    A = np.array(matrix, dtype=float)
    
    if decomposition_type == "lu":
        # LU decomposition with partial pivoting
        P, L, U = linalg.lu(A)
        return {
            "type": "LU",
            "P": P.tolist(),  # Permutation matrix
            "L": L.tolist(),  # Lower triangular
            "U": U.tolist(),  # Upper triangular
            "description": "A = P @ L @ U"
        }
    
    elif decomposition_type == "qr":
        # QR decomposition using Householder reflections
        Q, R = linalg.qr(A)
        return {
            "type": "QR",
            "Q": Q.tolist(),  # Orthogonal matrix
            "R": R.tolist(),  # Upper triangular
            "description": "A = Q @ R",
            "Q_is_orthogonal": True
        }
    
    elif decomposition_type == "cholesky":
        # Cholesky decomposition (for positive definite matrices)
        if A.shape[0] != A.shape[1]:
            raise ValueError("Cholesky requires a square matrix")
        
        try:
            L = linalg.cholesky(A, lower=True)
            return {
                "type": "Cholesky",
                "L": L.tolist(),  # Lower triangular
                "description": "A = L @ L.T",
                "positive_definite": True
            }
        except linalg.LinAlgError:
            return {
                "type": "Cholesky",
                "error": "Matrix is not positive definite",
                "positive_definite": False
            }
    
    elif decomposition_type == "svd":
        # Singular Value Decomposition
        U, s, Vt = linalg.svd(A, full_matrices=False)
        return {
            "type": "SVD",
            "U": U.tolist(),  # Left singular vectors
            "singular_values": s.tolist(),
            "Vt": Vt.tolist(),  # Right singular vectors (transposed)
            "description": "A = U @ diag(s) @ Vt",
            "rank": int(np.sum(s > 1e-10)),
            "condition_number": float(s[0] / s[-1]) if s[-1] > 1e-10 else float('inf')
        }
    
    elif decomposition_type == "eigen":
        # Eigenvalue decomposition
        if A.shape[0] != A.shape[1]:
            raise ValueError("Eigenvalue decomposition requires a square matrix")
        
        eigenvalues, eigenvectors = linalg.eig(A)
        
        # Separate real and complex parts
        eigenvalues_list = [
            {"real": float(val.real), "imag": float(val.imag)}
            for val in eigenvalues
        ]
        
        return {
            "type": "Eigenvalue",
            "eigenvalues": eigenvalues_list,
            "eigenvectors": eigenvectors.tolist(),
            "description": "A @ v = λ @ v",
            "has_complex_eigenvalues": bool(np.any(np.abs(eigenvalues.imag) > 1e-10))
        }
    
    else:
        raise ValueError(f"Unknown decomposition type: {decomposition_type}")


# ============================================================================
# Tool 4: Solve Linear System
# ============================================================================

def solve_linear_system(
    A: List[List[float]],
    b: List[float],
    method: str = "direct",
    tolerance: float = 1e-6,
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """
    Solve linear system Ax = b using direct or iterative methods.
    
    Args:
        A: Coefficient matrix (2D list)
        b: Right-hand side vector
        method: Solution method ('direct', 'jacobi', 'gauss_seidel', 'cg')
        tolerance: Convergence tolerance for iterative methods
        max_iterations: Maximum iterations for iterative methods
        
    Returns:
        Dictionary with solution and diagnostics
    """
    A_mat = np.array(A, dtype=float)
    b_vec = np.array(b, dtype=float)
    
    if A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError(f"Coefficient matrix must be square, got shape {A_mat.shape}")
    
    if A_mat.shape[0] != len(b_vec):
        raise ValueError(f"Matrix rows ({A_mat.shape[0]}) must match vector length ({len(b_vec)})")
    
    # Check condition number
    cond_number = float(np.linalg.cond(A_mat))
    
    if method == "direct":
        # Direct solver using LU decomposition
        try:
            x = linalg.solve(A_mat, b_vec)
            residual = np.linalg.norm(A_mat @ x - b_vec)
            
            return {
                "solution": x.tolist(),
                "method": "direct (LU)",
                "residual_norm": float(residual),
                "condition_number": cond_number,
                "converged": True
            }
        except linalg.LinAlgError as e:
            return {
                "error": str(e),
                "condition_number": cond_number,
                "suggestion": "Matrix may be singular or ill-conditioned"
            }
    
    elif method == "jacobi":
        # Jacobi iterative method
        x = np.zeros_like(b_vec)
        D = np.diag(np.diag(A_mat))
        R = A_mat - D
        
        for iteration in range(max_iterations):
            x_new = np.linalg.solve(D, b_vec - R @ x)
            
            if np.linalg.norm(x_new - x) < tolerance:
                residual = np.linalg.norm(A_mat @ x_new - b_vec)
                return {
                    "solution": x_new.tolist(),
                    "method": "Jacobi",
                    "iterations": iteration + 1,
                    "residual_norm": float(residual),
                    "converged": True
                }
            
            x = x_new
        
        return {
            "error": f"Failed to converge in {max_iterations} iterations",
            "method": "Jacobi",
            "suggestion": "Try direct method or increase max_iterations"
        }
    
    elif method == "gauss_seidel":
        # Gauss-Seidel iterative method
        x = np.zeros_like(b_vec)
        
        for iteration in range(max_iterations):
            x_old = x.copy()
            
            for i in range(len(x)):
                sigma = np.dot(A_mat[i, :i], x[:i]) + np.dot(A_mat[i, i+1:], x_old[i+1:])
                x[i] = (b_vec[i] - sigma) / A_mat[i, i]
            
            if np.linalg.norm(x - x_old) < tolerance:
                residual = np.linalg.norm(A_mat @ x - b_vec)
                return {
                    "solution": x.tolist(),
                    "method": "Gauss-Seidel",
                    "iterations": iteration + 1,
                    "residual_norm": float(residual),
                    "converged": True
                }
        
        return {
            "error": f"Failed to converge in {max_iterations} iterations",
            "method": "Gauss-Seidel",
            "suggestion": "Try direct method or increase max_iterations"
        }
    
    elif method == "cg":
        # Conjugate Gradient (for symmetric positive definite matrices)
        if not np.allclose(A_mat, A_mat.T):
            return {
                "error": "Conjugate Gradient requires a symmetric matrix",
                "suggestion": "Use direct, jacobi, or gauss_seidel method"
            }
        
        x = np.zeros_like(b_vec)
        r = b_vec - A_mat @ x
        p = r.copy()
        rs_old = np.dot(r, r)
        
        for iteration in range(max_iterations):
            Ap = A_mat @ p
            alpha = rs_old / np.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = np.dot(r, r)
            
            if np.sqrt(rs_new) < tolerance:
                residual = np.linalg.norm(A_mat @ x - b_vec)
                return {
                    "solution": x.tolist(),
                    "method": "Conjugate Gradient",
                    "iterations": iteration + 1,
                    "residual_norm": float(residual),
                    "converged": True
                }
            
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        
        return {
            "error": f"Failed to converge in {max_iterations} iterations",
            "method": "Conjugate Gradient",
            "suggestion": "Increase max_iterations or try direct method"
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Tool 5: Vector Operations
# ============================================================================

def vector_operations(
    operation: str,
    vector_a: List[float],
    vector_b: List[float] = None
) -> Dict[str, Any]:
    """
    Perform vector operations: dot product, cross product, norms, projections.
    
    Args:
        operation: Operation to perform ('dot', 'cross', 'norm', 'normalize', 'angle', 'projection')
        vector_a: First vector
        vector_b: Second vector (for binary operations)
        
    Returns:
        Dictionary with operation result
    """
    a = np.array(vector_a, dtype=float)
    
    if operation == "norm":
        norm_type = vector_b[0] if vector_b and len(vector_b) > 0 else 2  # Default L2 norm
        
        if norm_type == 1:
            result = float(np.linalg.norm(a, ord=1))
            norm_name = "L1 (Manhattan)"
        elif norm_type == 2:
            result = float(np.linalg.norm(a, ord=2))
            norm_name = "L2 (Euclidean)"
        elif norm_type == float('inf'):
            result = float(np.linalg.norm(a, ord=np.inf))
            norm_name = "L∞ (Maximum)"
        else:
            result = float(np.linalg.norm(a, ord=norm_type))
            norm_name = f"L{norm_type}"
        
        return {"norm": result, "norm_type": norm_name}
    
    elif operation == "normalize":
        norm = np.linalg.norm(a)
        if norm < 1e-10:
            raise ValueError("Cannot normalize zero vector")
        
        normalized = (a / norm).tolist()
        return {"normalized": normalized, "original_norm": float(norm)}
    
    # Binary operations
    if vector_b is None:
        raise ValueError(f"Operation '{operation}' requires vector_b")
    
    b = np.array(vector_b, dtype=float)
    
    if operation == "dot":
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same length: {len(a)} vs {len(b)}")
        
        result = float(np.dot(a, b))
        return {"dot_product": result}
    
    elif operation == "cross":
        if len(a) != 3 or len(b) != 3:
            raise ValueError("Cross product requires 3D vectors")
        
        result = np.cross(a, b).tolist()
        magnitude = float(np.linalg.norm(np.cross(a, b)))
        return {"cross_product": result, "magnitude": magnitude}
    
    elif operation == "angle":
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same length: {len(a)} vs {len(b)}")
        
        dot_prod = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            raise ValueError("Cannot compute angle with zero vector")
        
        cos_angle = dot_prod / (norm_a * norm_b)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        
        angle_rad = float(np.arccos(cos_angle))
        angle_deg = float(np.degrees(angle_rad))
        
        return {
            "angle_radians": angle_rad,
            "angle_degrees": angle_deg,
            "cosine": float(cos_angle)
        }
    
    elif operation == "projection":
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same length: {len(a)} vs {len(b)}")
        
        # Project a onto b
        norm_b_sq = np.dot(b, b)
        if norm_b_sq < 1e-10:
            raise ValueError("Cannot project onto zero vector")
        
        proj = (np.dot(a, b) / norm_b_sq) * b
        return {
            "projection": proj.tolist(),
            "magnitude": float(np.linalg.norm(proj))
        }
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


# ============================================================================
# Tool 6: Least Squares Fit
# ============================================================================

def least_squares_fit(
    X: List[List[float]],
    y: List[float],
    method: str = "ols",
    alpha: float = 0.0
) -> Dict[str, Any]:
    """
    Perform least squares regression with various methods.
    
    Args:
        X: Design matrix (features)
        y: Target vector
        method: Regression method ('ols', 'wls', 'ridge', 'lasso')
        alpha: Regularization parameter for Ridge/Lasso
        
    Returns:
        Dictionary with regression coefficients and diagnostics
    """
    X_mat = np.array(X, dtype=float)
    y_vec = np.array(y, dtype=float)
    
    if X_mat.shape[0] != len(y_vec):
        raise ValueError(f"Number of samples must match: X has {X_mat.shape[0]}, y has {len(y_vec)}")
    
    n, p = X_mat.shape
    
    if method == "ols":
        # Ordinary Least Squares using QR decomposition (numerically stable)
        coefficients, residuals, rank, s = linalg.lstsq(X_mat, y_vec)
        
        # Calculate diagnostics
        y_pred = X_mat @ coefficients
        residuals_vec = y_vec - y_pred
        
        # R-squared
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        ss_res = np.sum(residuals_vec ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        
        # RMSE
        rmse = float(np.sqrt(ss_res / n))
        
        # Condition number
        cond_number = float(np.linalg.cond(X_mat))
        
        return {
            "coefficients": coefficients.tolist(),
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(adj_r_squared),
            "rmse": rmse,
            "residuals": residuals_vec.tolist(),
            "condition_number": cond_number,
            "method": "OLS"
        }
    
    elif method == "ridge":
        # Ridge Regression (L2 regularization)
        if alpha <= 0:
            raise ValueError("Ridge regression requires alpha > 0")
        
        # Add regularization to normal equations
        XtX = X_mat.T @ X_mat + alpha * np.eye(p)
        Xty = X_mat.T @ y_vec
        coefficients = linalg.solve(XtX, Xty)
        
        y_pred = X_mat @ coefficients
        residuals_vec = y_vec - y_pred
        
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        ss_res = np.sum(residuals_vec ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        rmse = float(np.sqrt(ss_res / n))
        
        return {
            "coefficients": coefficients.tolist(),
            "r_squared": float(r_squared),
            "rmse": rmse,
            "residuals": residuals_vec.tolist(),
            "alpha": alpha,
            "method": "Ridge"
        }
    
    elif method == "wls":
        # Weighted Least Squares (weights in alpha parameter as list)
        # For simplicity, treat as OLS if no weights provided
        return least_squares_fit(X, y, method="ols", alpha=0.0)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ols', 'ridge', or 'wls'")


# ============================================================================
# Tool Definitions
# ============================================================================

LINEAR_ALGEBRA_TOOLS = [
    Tool(
        name="matrix_operations",
        description="Perform matrix operations: add, subtract, multiply, transpose, trace, determinant",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "transpose", "trace", "determinant"],
                    "description": "Operation to perform"
                },
                "matrix_a": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "First matrix (2D array)"
                },
                "matrix_b": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Second matrix (optional, required for binary operations)"
                }
            },
            "required": ["operation", "matrix_a"]
        }
    ),
    Tool(
        name="matrix_inverse",
        description="Compute matrix inverse using LU, Cholesky, or SVD methods with conditioning analysis",
        inputSchema={
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Square matrix to invert"
                },
                "method": {
                    "type": "string",
                    "enum": ["lu", "cholesky", "svd"],
                    "description": "Inversion method (default: lu)"
                },
                "regularization": {
                    "type": "number",
                    "description": "Ridge regularization parameter (default: 0.0)"
                }
            },
            "required": ["matrix"]
        }
    ),
    Tool(
        name="matrix_decomposition",
        description="Perform matrix decomposition: LU, QR, Cholesky, SVD, or Eigenvalue",
        inputSchema={
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Input matrix"
                },
                "decomposition_type": {
                    "type": "string",
                    "enum": ["lu", "qr", "cholesky", "svd", "eigen"],
                    "description": "Type of decomposition (default: lu)"
                }
            },
            "required": ["matrix"]
        }
    ),
    Tool(
        name="solve_linear_system",
        description="Solve linear system Ax=b using direct or iterative methods",
        inputSchema={
            "type": "object",
            "properties": {
                "A": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Coefficient matrix"
                },
                "b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Right-hand side vector"
                },
                "method": {
                    "type": "string",
                    "enum": ["direct", "jacobi", "gauss_seidel", "cg"],
                    "description": "Solution method (default: direct)"
                },
                "tolerance": {
                    "type": "number",
                    "description": "Convergence tolerance (default: 1e-6)"
                },
                "max_iterations": {
                    "type": "number",
                    "description": "Maximum iterations (default: 1000)"
                }
            },
            "required": ["A", "b"]
        }
    ),
    Tool(
        name="vector_operations",
        description="Perform vector operations: dot product, cross product, norms, angles, projections",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["dot", "cross", "norm", "normalize", "angle", "projection"],
                    "description": "Operation to perform"
                },
                "vector_a": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "First vector"
                },
                "vector_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Second vector (optional, required for binary operations)"
                }
            },
            "required": ["operation", "vector_a"]
        }
    ),
    Tool(
        name="least_squares_fit",
        description="Perform least squares regression (OLS, Ridge) with diagnostics",
        inputSchema={
            "type": "object",
            "properties": {
                "X": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Design matrix (features)"
                },
                "y": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Target vector"
                },
                "method": {
                    "type": "string",
                    "enum": ["ols", "wls", "ridge"],
                    "description": "Regression method (default: ols)"
                },
                "alpha": {
                    "type": "number",
                    "description": "Regularization parameter for Ridge (default: 0.0)"
                }
            },
            "required": ["X", "y"]
        }
    )
]


# ============================================================================
# Tool Handlers
# ============================================================================

async def handle_matrix_operations(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle matrix_operations tool calls."""
    try:
        result = matrix_operations(
            operation=arguments["operation"],
            matrix_a=arguments["matrix_a"],
            matrix_b=arguments.get("matrix_b")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in matrix_operations: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_matrix_inverse(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle matrix_inverse tool calls."""
    try:
        result = matrix_inverse(
            matrix=arguments["matrix"],
            method=arguments.get("method", "lu"),
            regularization=arguments.get("regularization", 0.0)
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in matrix_inverse: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_matrix_decomposition(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle matrix_decomposition tool calls."""
    try:
        result = matrix_decomposition(
            matrix=arguments["matrix"],
            decomposition_type=arguments.get("decomposition_type", "lu")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in matrix_decomposition: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_solve_linear_system(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle solve_linear_system tool calls."""
    try:
        result = solve_linear_system(
            A=arguments["A"],
            b=arguments["b"],
            method=arguments.get("method", "direct"),
            tolerance=arguments.get("tolerance", 1e-6),
            max_iterations=arguments.get("max_iterations", 1000)
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in solve_linear_system: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_vector_operations(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle vector_operations tool calls."""
    try:
        result = vector_operations(
            operation=arguments["operation"],
            vector_a=arguments["vector_a"],
            vector_b=arguments.get("vector_b")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in vector_operations: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_least_squares_fit(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle least_squares_fit tool calls."""
    try:
        result = least_squares_fit(
            X=arguments["X"],
            y=arguments["y"],
            method=arguments.get("method", "ols"),
            alpha=arguments.get("alpha", 0.0)
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in least_squares_fit: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )
