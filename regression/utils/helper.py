import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from prettytable import PrettyTable

"""
ERA5 data helper functions
"""


def get_inner_circle_indices(n, flat=False):
    '''
    Input: n - int - size of square
    Ouput: Ind - torch.tensor
    '''
    x_axis = torch.linspace(start=0, end=n - 1, steps=n)
    y_axis = torch.linspace(start=0, end=n - 1, steps=n)
    X1, X2 = torch.meshgrid(x_axis, y_axis)
    Ind = torch.stack((X1, X2), 2)
    Ind = Ind[torch.norm(Ind - (n - 1) / 2, dim=2) < (n - 1) / 2].long()
    if flat:
        Ind = n * Ind[:, 0] + Ind[:, 1]
    return (Ind)


def origin_plot_inference_2d(X_Context, Y_Context, X_Target=None, Y_Target=None, Predict=None, Cov_Mat=None, title="",
                      size_scale=2, ellip_scale=0.8, quiver_scale=15, plot_points=False):
    '''
    Inputs: X_Context,Y_Context: torch.tensor - shape (n_context_points,2) - given context set
            X_Target,Y_Target: torch.tensor - shape (n_target_points,2) - target locations and outputs vectors
            Predict: torch.tensor - shape (n_target_points,2) - target predictions (i.e. predicting Y_Target)
            Cov_Mat: torch.tensor - shape (n_target_points,2,2) - set of covariance matrices
                                  or - shape (n_target_points,2) - set of variances
            title: string -  suptitle
    Outputs: None - plots the above tensors in two plots: one plots the means against the context sets, the other one the covariances/variances
    '''
    # Create plots:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(size_scale * 10, size_scale * 5))
    plt.gca().set_aspect('equal', adjustable='box')
    fig.subplots_adjust(wspace=0.4)
    # Set titles:
    fig.suptitle(title)
    ax[0].set_title("Context set and predictions")
    ax[1].set_title("Variances")

    # Plot context set in blue:
    if plot_points:
        ax[0].scatter(X_Context[:, 0], X_Context[:, 1], color='black')
    if X_Context is not None and Y_Context is not None:
        ax[0].quiver(X_Context[:, 0], X_Context[:, 1], Y_Context[:, 0], Y_Context[:, 1],
                     color='black', pivot='mid', label='Context set', scale=quiver_scale)

    # Plot ground truth in red if given:
    if Y_Target is not None and X_Target is not None:
        ax[0].quiver(X_Target[:, 0], X_Target[:, 1], Y_Target[:, 0], Y_Target[:, 1],
                     color='blue', pivot='mid', label='Ground truth', scale=quiver_scale)

    # Plot predicted means in green:
    if Predict is not None and X_Target is not None:
        ax[0].quiver(X_Target[:, 0], X_Target[:, 1], Predict[:, 0], Predict[:, 1], color='red', pivot='mid',
                     label='Predictions', scale=quiver_scale)

    leg = ax[0].legend(bbox_to_anchor=(1., 1.))

    if X_Target is not None and Cov_Mat is not None:
        # Get window limites for plot and set window for second plot:
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].scatter(X_Context[:, 0], X_Context[:, 1], color='black', label='Context Points', marker='+')
        # Go over all target points and plot ellipse of continour lines of density of distributions:
        for j in range(X_Target.size(0)):
            # Get covarinace matrix:
            A = Cov_Mat[j]
            if len(A.size()) == 1:
                A = torch.diag(A)

            # Decompose A:
            eigen_decomp = torch.eig(A, eigenvectors=True)
            # Get the eigenvector corresponding corresponding to the largest eigenvalue:
            u = eigen_decomp[1][:, 0]

            # Get the angle of the ellipse in degrees:
            alpha = 360 * torch.atan(u[1] / u[0]) / (2 * math.pi)

            # Get the width and height of the ellipses (eigenvalues of A):
            D = torch.sqrt(eigen_decomp[0][:, 0])

            # Plot the Ellipse:
            E = Ellipse(xy=X_Target[j,].numpy(), width=ellip_scale * D[0], height=ellip_scale * D[1], angle=alpha)
            ax[1].add_patch(E)

        # Create a legend:
        blue_ellipse = Ellipse(color='c', label='Confid. ellip.', xy=0, width=1, height=1)
        ax[1].legend(handles=[blue_ellipse])
        leg = ax[1].legend(bbox_to_anchor=(1., 1.))


def count_parameters(model, print_table=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if print_table:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params


"""
Plot Helper Function
"""


#
# Tool to plot context set, ground truth for target and predictions for target in one plot:
# def plot_inference_2d(X_Context, Y_Context, X_Target=None, Y_Target=None, Predict=None, Cov_Mat=None, title="",
#                       size_scale=2, ellip_scale=0.8, quiver_scale=15, plot_points=False,out_path=''):
#
#     print("this is the function ")
#     '''
#     Inputs: X_Context,Y_Context: torch.tensor - shape (n_context_points,2) - given context set
#             X_Target,Y_Target: torch.tensor - shape (n_target_points,2) - target locations and outputs vectors
#             Predict: torch.tensor - shape (n_target_points,2) - target predictions (i.e. predicting Y_Target)
#             Cov_Mat: torch.tensor - shape (n_target_points,2,2) - set of covariance matrices
#                                   or - shape (n_target_points,2) - set of variances
#             title: string -  suptitle
#     Outputs: None - plots the above tensors in two plots: one plots the means against the context sets, the other one the covariances/variances
#     '''
#     print("X_Context shape:", X_Context.shape)
#     print("Y_Context shape:", Y_Context.shape)
#     print("X_Target shape:", X_Target.shape)
#     print("Y_Target shape:", Y_Target.shape)
#     print("Predict shape:", Predict.shape)
#     print("Cov_Mat shape:", Cov_Mat.shape)
#
#
#     # Create plots:
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(size_scale * 10, size_scale * 5))
#     plt.gca().set_aspect('equal', adjustable='box')
#     fig.subplots_adjust(wspace=0.4)
#     # Set titles:
#     fig.suptitle(title)
#     ax[0].set_title("Context set and predictions")
#     ax[1].set_title("Variances")
#
#     # Plot context set in blue:
#     if plot_points:
#         ax[0].scatter(X_Context[:, 0], X_Context[:, 1], color='black')
#     if X_Context is not None and Y_Context is not None:
#         ax[0].quiver(X_Context[:, 0], X_Context[:, 1], Y_Context[:, 0], Y_Context[:, 1],
#                      color='black', pivot='mid', label='Context set', scale=quiver_scale)
#
#     # Plot ground truth in red if given:
#     if Y_Target is not None and X_Target is not None:
#         ax[0].quiver(X_Target[:, 0], X_Target[:, 1], Y_Target[:, 0], Y_Target[:, 1],
#                      color='blue', pivot='mid', label='Ground truth', scale=quiver_scale)
#
#     # Plot predicted means in green:
#     if Predict is not None and X_Target is not None:
#         ax[0].quiver(X_Target[:, 0], X_Target[:, 1], Predict[:, 0], Predict[:, 1], color='red', pivot='mid',
#                      label='Predictions', scale=quiver_scale)
#
#     leg = ax[0].legend(bbox_to_anchor=(1., 1.))
#
#     if X_Target is not None and Cov_Mat is not None:
#         print("Cov_Mat",Cov_Mat.shape)
#         # Get window limites for plot and set window for second plot:
#         ax[1].set_xlim(ax[0].get_xlim())
#         ax[1].set_ylim(ax[0].get_ylim())
#         ax[1].scatter(X_Context[:, 0], X_Context[:, 1], color='black', label='Context Points', marker='+')
#         # Go over all target points and plot ellipse of continour lines of density of distributions:
#         for j in range(X_Target.size(0)):
#             # Get covarinace matrix:
#             A = Cov_Mat[j]
#             if len(A.size()) == 1:
#                 A = torch.diag(A)
#
#             # print("A shape:", A.shape)
#             # Decompose A:
#             # eigen_decomp = torch.eig(A, eigenvectors=True)
#             eigen_decomp = torch.linalg.eig(A)
#             # Get the eigenvector corresponding corresponding to the largest eigenvalue:
#             u = eigen_decomp[1][:, 0]
#
#             # Get the angle of the ellipse in degrees:
#             alpha = 360 * torch.atan(u[1] / u[0]) / (2 * math.pi)
#
#             # Get the width and height of the ellipses (eigenvalues of A):
#             # D = torch.sqrt(eigen_decomp[0][:, 0])
#             D = torch.sqrt(eigen_decomp[0].real)
#
#             # Plot the Ellipse:
#             E = Ellipse(xy=X_Target[j,].numpy(), width=ellip_scale * D[0], height=ellip_scale * D[1], angle=alpha)
#             ax[1].add_patch(E)
#
#         # Create a legend:
#         blue_ellipse = Ellipse(color='c', label='Confid. ellip.', xy=0, width=1, height=1)
#         ax[1].legend(handles=[blue_ellipse])
#         leg = ax[1].legend(bbox_to_anchor=(1., 1.))
#
#     plot_name = f"{out_path}/inference_plot.png"
#     plt.savefig(plot_name)
#     plt.show()
#     print(f"Plot saved at {plot_name}")
#
#     def plot_context_target(self, X_Context, Y_Context, X_Target, Y_Target=None, title=""):
#         '''
#             Inputs: X_Context, Y_Context, X_Target: torch.tensor - shape (batch_size,n_context/n_target,2)
#                     Y_Target: torch.tensor - shape (n_context_points,2) - ground truth
#             Output: None - plots predictions
#
#         '''
#         # Get predictions:
#         Means, Covs = self.forward(X_Context, Y_Context, X_Target)
#         # Plot predictions against ground truth:
#         for i in range(X_Context.size(0)):
#             plot_inference_2d(X_Context[i], Y_Context[i], X_Target[i], Y_Target[i], Predict=Means[i].detach(),
#                                        Cov_Mat=Covs[i].detach(), title=title)


def plot_inference_2d(X_Context, Y_Context, X_Target=None, Y_Target=None, Predict=None, Cov_Mat=None, title="",
                      size_scale=2, ellip_scale=0.8, quiver_scale=15, plot_points=False, out_path=''):
    '''
    Inputs:
        X_Context, Y_Context: torch.tensor - shape (n_context_points, 2) - given context set
        X_Target, Y_Target: torch.tensor - shape (n_target_points, 2) - target locations and outputs vectors
        Predict: torch.tensor - shape (n_target_points, 2) - target predictions (i.e. predicting Y_Target)
        Cov_Mat: torch.tensor - shape (n_target_points, 2, 2) - set of covariance matrices
                              or - shape (n_target_points, 2) - set of variances
        title: string - suptitle
    Outputs: None - plots the above tensors in two plots: one plots the means against the context sets,
                     the other one the covariances/variances
    '''

    # Print shapes for debugging
    print("X_Context shape:", X_Context.shape)
    print("Y_Context shape:", Y_Context.shape)
    print("X_Target shape:", X_Target.shape if X_Target is not None else "None")
    print("Y_Target shape:", Y_Target.shape if Y_Target is not None else "None")
    print("Predict shape:", Predict.shape if Predict is not None else "None")
    print("Cov_Mat shape:", Cov_Mat.shape if Cov_Mat is not None else "None")

    # Create plots:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(size_scale * 10, size_scale * 5))
    plt.gca().set_aspect('equal', adjustable='box')
    fig.subplots_adjust(wspace=0.4)

    # Set titles:
    num_context_points = X_Context.size(0)
    num_predictions = Predict.size(0) if Predict is not None else 0
    ax[0].set_title(
        f"Context set and predictions\nContext: {num_context_points}, Predictions: {num_predictions}")

    # Plot context set in black:
    if plot_points:
        ax[0].scatter(X_Context[:, 0], X_Context[:, 1], color='black', label='Context Points')
    if X_Context is not None and Y_Context is not None:
        ax[0].quiver(X_Context[:, 0], X_Context[:, 1], Y_Context[:, 0], Y_Context[:, 1],
                     color='black', pivot='mid', label='Context set', scale=quiver_scale)

    # Plot ground truth in blue if given:
    if Y_Target is not None and X_Target is not None:
        ax[0].quiver(X_Target[:, 0], X_Target[:, 1], Y_Target[:, 0], Y_Target[:, 1],
                     color='blue', pivot='mid', label='Ground Truth', scale=quiver_scale)

    # Plot predicted means in red:
    if Predict is not None and X_Target is not None:
        ax[0].quiver(X_Target[:, 0], X_Target[:, 1], Predict[:, 0], Predict[:, 1], color='red', pivot='mid',
                     label='Predictions', scale=quiver_scale)

    leg = ax[0].legend(bbox_to_anchor=(1., 1.))

    if X_Target is not None and Cov_Mat is not None:
        # Get window limits for plot and set window for the second plot:
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].scatter(X_Context[:, 0], X_Context[:, 1], color='black', label='Context Points', marker='+')

        # Go over all target points and plot ellipses:
        for j in range(X_Target.size(0)):
            # Get covariance matrix:
            A = Cov_Mat[j]
            if len(A.size()) == 1:
                A = torch.diag(A)

            # Check if covariance matrix is valid
            if A.size(0) != 2 or A.size(1) != 2:
                print(f"Invalid covariance matrix at index {j}: {A}")
                continue

            # Decompose A:
            eigen_decomp = torch.linalg.eig(A)
            u = eigen_decomp[1][:, 0].real  # Eigenvector
            alpha = torch.atan2(u[1], u[0]) * (180 / math.pi)  # Angle in degrees
            D = torch.sqrt(eigen_decomp[0].real)  # Eigenvalues (width and height)

            # Plot the Ellipse:
            E = Ellipse(xy=X_Target[j].numpy(), width=ellip_scale * D[0].item(), height=ellip_scale * D[1].item(),
                        angle=alpha)
            ax[1].add_patch(E)

        # Create a legend using the actual ellipses
        blue_ellipse = Ellipse(xy=(0, 0), color='c', label='Confid. ellip.', width=1, height=1)
        ax[1].legend(handles=[blue_ellipse], bbox_to_anchor=(1., 1.))

    # Save and show the plot
    plot_name = f"{out_path}/inference_plot.png"
    plt.savefig(plot_name)
    plt.show()
    print(f"Plot saved at {plot_name}")



def plot_temperature(xc, yc, xt, yt, py, save_path):

    print("xc shape:", xc.shape)
    print("yc shape:", yc.shape)
    print("xt shape:", xt.shape)
    print("yt shape:", yt.shape)
    print("py shape:", py.shape)

    """
    Plot temperature predictions and ground truth using grid points.

    Parameters:
    xc: Tensor containing context points (shape: [16, 727, 2]) with spatial coordinates
    yc: Tensor containing context ground truth (shape: [16, 727, 4])
    xt: Tensor containing target points (shape: [16, 518, 2]) with spatial coordinates
    yt: Tensor containing ground truth targets (shape: [16, 518, 4])
    py: Tensor containing predicted values (shape: [16, 518, 4])
    save_path: Path where the plot will be saved
    """

    # Extract temperature (variable 1)
    # Extract temperature (variable 1)
    temp_xc = yc[:, 1]  # Context ground truth temperature
    temp_yc = yt[:, 1]  # Target ground truth temperature
    temp_py = py[:, 1]   # Predicted temperature

    # Determine the min and max temperature values for consistent scaling
    vmin = min(temp_xc.min(), temp_yc.min(), temp_py.min())
    vmax = max(temp_xc.max(), temp_yc.max(), temp_py.max())

    # Create a grid of points for context and target
    xc_points = xc.numpy()  # Convert tensor to numpy array
    xt_points = xt.numpy()

    # Create a new figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # Plot context ground truth temperature
    scatter1 = axs[0].scatter(xc_points[ :, 0], xc_points[ :, 1], c=temp_xc.flatten(), cmap='coolwarm', marker='o', s=5)
    # scatter1.set_clim(vmin, vmax)
    axs[0].set_title(f'Context Ground Truth Temperature. Context points: {xc_points.shape[0]}')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    fig.colorbar(scatter1, ax=axs[0], label='Temperature (°C)')

    # Plot target ground truth temperature
    scatter2 = axs[1].scatter(xt_points[ :, 0], xt_points[ :, 1], c=temp_yc.flatten(), cmap='coolwarm', marker='o', s=5)
    # scatter2.set_clim(vmin, vmax)
    axs[1].set_title(f'Target Ground Truth Temperature. Target points: {xt_points.shape[0]}')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    fig.colorbar(scatter2, ax=axs[1], label='Temperature (°C)')

    # Plot predicted temperature
    scatter3 = axs[2].scatter(xt_points[ :, 0], xt_points[ :, 1], c=temp_py.flatten(), cmap='coolwarm', marker='o', s=5)
    # scatter3.set_clim(vmin, vmax)
    axs[2].set_title(f'Predicted Temperature. Target points: {xt_points.shape[0]}')
    axs[2].set_xlabel('Longitude')
    axs[2].set_ylabel('Latitude')
    fig.colorbar(scatter3, ax=axs[2], label='Temperature (°C)')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved at {save_path}")
