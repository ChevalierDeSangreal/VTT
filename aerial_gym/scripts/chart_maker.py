from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

def main_test_height():
    """
    In order to compare changed test with original test.
    """
    # print("??????????")
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_height__small_tar') 
    ea.Reload()
    # print(ea.scalars.Keys())
    # print("??????????")
    batch_size = 8
    
    data = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss', 'Speed']:
            data_values = [j.value for j in ea.scalars.Items(f'{data_name}{i}')]
            data[data_name].append(data_values)
            
            
    ea2 = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_main') 
    ea2.Reload()
    
    data2 = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss', 'Speed']:
            data_values = [j.value for j in ea2.scalars.Items(f'{data_name}{i}')]
            data2[data_name].append(data_values)
    
    # Create a plot for each type of data
    for key, val in data.items():
        plt.figure(figsize=(12, 8))
        for idx, batch_data in enumerate(val):
            if idx == 0:
                plt.plot(batch_data, label=f'{key} {idx}', alpha=0.3)  # Original data with moderate transparency
                average_curve = np.array(batch_data)
            else:
                plt.plot(batch_data, label=f'{key} {idx}', alpha=0.3)  # Original data with low transparency
                average_curve += np.array(batch_data)
                
        val2 = data2[key]
        for idx, batch_data in enumerate(val2):
            if idx == 0:
                average_curve2 = np.array(batch_data)
            else:
                average_curve2 += np.array(batch_data)
                
        average_curve /= len(val)  # Calculate the average curve
        average_curve2 /= len(val)  # Calculate the average curve
        plt.plot(average_curve, label=f'Ground Changed {key} (Average)', linewidth=2, color='#0000CD', alpha=0.8)  # Plot the average curve with higher transparency
        plt.plot(average_curve2, label=f'Standard {key} (Average)', linewidth=2, color='#FF0000', alpha=0.8)  # Plot the average curve with higher transparency
        
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'{key} Data')
        plt.savefig(f'/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/test_height__small_tar/{key}_plot.png')  # Save the plot to a file with the key name
        plt.close()
    print("Complete!")


def main_test_moving():
    """
    In order to compare changed test with original test.
    """
    # print("??????????")
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_moving__1.5__h') 
    ea.Reload()
    # print(ea.scalars.Keys())
    # print("??????????")
    batch_size = 8
    
    data = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss', 'Speed', "Height"]:
            data_values = [j.value for j in ea.scalars.Items(f'{data_name}{i}')]
            data[data_name].append(data_values)
            
            
    ea2 = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_main__h') 
    ea2.Reload()
    
    data2 = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss', 'Speed', "Height"]:
            data_values = [j.value for j in ea2.scalars.Items(f'{data_name}{i}')]
            data2[data_name].append(data_values)
    
    # Create a plot for each type of data
    for key, val in data.items():
        plt.figure(figsize=(12, 8))
        for idx, batch_data in enumerate(val):
            if idx == 0:
                plt.plot(batch_data, label=f'{key} {idx}', alpha=0.3)  # Original data with moderate transparency
                average_curve = np.array(batch_data)
            else:
                plt.plot(batch_data, label=f'{key} {idx}', alpha=0.3)  # Original data with low transparency
                average_curve += np.array(batch_data)
                
        val2 = data2[key]
        for idx, batch_data in enumerate(val2):
            if idx == 0:
                average_curve2 = np.array(batch_data)
            else:
                average_curve2 += np.array(batch_data)
                
        average_curve /= len(val)  # Calculate the average curve
        average_curve2 /= len(val)  # Calculate the average curve
        plt.plot(average_curve, label=f'Ground Changed {key} (Average)', linewidth=2, color='#0000CD', alpha=0.8)  # Plot the average curve with higher transparency
        plt.plot(average_curve2, label=f'Standard {key} (Average)', linewidth=2, color='#FF0000', alpha=0.8)  # Plot the average curve with higher transparency
        
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'{key}')
        plt.savefig(f'/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/test_moving__1.5/{key}_plot.png')  # Save the plot to a file with the key name
        plt.close()
    print("Complete!")
    
def main_test_movingVer2():
    # print("??????????")
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_moving__3__h') 
    ea.Reload()
    # print(ea.scalars.Keys())
    # print("??????????")
    batch_size = 8
    
    data = defaultdict(list)
    
    # Collect data for each type
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Speed', "Height"]:
            data_values = [j.value for j in ea.scalars.Items(f'{data_name}{i}')]
            data[data_name].append(data_values)
    
    ea2 = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_main__h') 
    ea2.Reload()
    
    data2 = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Speed', "Height"]:
            data_values = [j.value for j in ea2.scalars.Items(f'{data_name}{i}')]
            data2[data_name].append(data_values)
    
    # Plot each type of data separately
    plt.figure(figsize=(10, 7))  # Set the figure size wider and taller based on the number of data types
    for idx, (key, val) in enumerate(data.items()):
        plt.subplot(len(data), 1, idx + 1)  # Vertical arrangement with each data type in a separate row
        average_curve = np.zeros(len(val[0]))  # Initialize an array to store the average curve
        for batch_data in val:
            plt.plot(batch_data, alpha=0.3)  # Original data with moderate transparency
            average_curve += np.array(batch_data)
            
        val2 = data2[key]
        for i, batch_data2 in enumerate(val2):
            if i == 0:
                average_curve2 = np.array(batch_data2)
            else:
                average_curve2 += np.array(batch_data2)
        average_curve /= len(val)  # Calculate the average curve
        average_curve2 /= len(val)
        plt.plot(average_curve, label='Average', linewidth=2, color='#0000CD', alpha=0.8)  # Plot the average curve with higher transparency
        plt.plot(average_curve2, label=f'Standard Average', linewidth=2, color='#FF0000', alpha=0.8)  # Plot the average curve with higher transparency
        
        plt.legend()
        
        plt.xlabel('Step')
        if key == "Horizon Distance" or key == "Vertical Distance" or key == "Height":
            plt.ylabel(key + '(m)')
        elif key == "Speed":
            plt.ylabel(key + '(m/s)')
        else:
            plt.ylabel(key)
        # plt.title(f'{key}')

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/test_moving__3.png')  # Save the combined plot to a single file
    plt.close()
    print("Complete!")

def main_test_ground():
    """
    In order to compare changed test with original test.
    """
    # print("??????????")
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_ground') 
    ea.Reload()
    # print(ea.scalars.Keys())
    # print("??????????")
    batch_size = 8
    
    data = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss', 'Speed']:
            data_values = [j.value for j in ea.scalars.Items(f'{data_name}{i}')]
            data[data_name].append(data_values)
            
            
    ea2 = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_main') 
    ea2.Reload()
    
    data2 = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss', 'Speed']:
            data_values = [j.value for j in ea2.scalars.Items(f'{data_name}{i}')]
            data2[data_name].append(data_values)
    
    # Create a plot for each type of data
    for key, val in data.items():
        plt.figure(figsize=(12, 8))
        for idx, batch_data in enumerate(val):
            if idx == 0:
                plt.plot(batch_data, label=f'{key} {idx}', alpha=0.3)  # Original data with moderate transparency
                average_curve = np.array(batch_data)
            else:
                plt.plot(batch_data, label=f'{key} {idx}', alpha=0.3)  # Original data with low transparency
                average_curve += np.array(batch_data)
                
        val2 = data2[key]
        for idx, batch_data in enumerate(val2):
            if idx == 0:
                average_curve2 = np.array(batch_data)
            else:
                average_curve2 += np.array(batch_data)
                
        average_curve /= len(val)  # Calculate the average curve
        average_curve2 /= len(val)  # Calculate the average curve
        plt.plot(average_curve, label=f'Ground Changed {key} (Average)', linewidth=2, color='#0000CD', alpha=0.8)  # Plot the average curve with higher transparency
        plt.plot(average_curve2, label=f'Standard {key} (Average)', linewidth=2, color='#FF0000', alpha=0.8)  # Plot the average curve with higher transparency
        
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'{key} Data')
        plt.savefig(f'/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/test_ground/{key}_plot.png')  # Save the plot to a file with the key name
        plt.close()
    print("Complete!")
    
def test_heightVer2():
    # print("??????????")
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_height__h') 
    ea.Reload()
    # print(ea.scalars.Keys())
    # print("??????????")
    batch_size = 8
    
    data = defaultdict(list)
    
    # Collect data for each type
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Height', 'Speed', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss']:
            data_values = [j.value for j in ea.scalars.Items(f'{data_name}{i}')]
            data[data_name].append(data_values)
    
    ea2 = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_main__h') 
    ea2.Reload()
    
    data2 = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Height', 'Speed', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss']:
            data_values = [j.value for j in ea2.scalars.Items(f'{data_name}{i}')]
            data2[data_name].append(data_values)
    
    # Plot each type of data separately
    plt.figure(figsize=(10, 20))  # Set the figure size wider and taller based on the number of data types
    for idx, (key, val) in enumerate(data.items()):
        plt.subplot(len(data), 1, idx + 1)  # Vertical arrangement with each data type in a separate row
        average_curve = np.zeros(len(val[0]))  # Initialize an array to store the average curve
        for batch_data in val:
            plt.plot(batch_data, alpha=0.3)  # Original data with moderate transparency
            average_curve += np.array(batch_data)
            
        val2 = data2[key]
        for i, batch_data2 in enumerate(val2):
            if i == 0:
                average_curve2 = np.array(batch_data2)
            else:
                average_curve2 += np.array(batch_data2)
        average_curve /= len(val)  # Calculate the average curve
        average_curve2 /= len(val)
        plt.plot(average_curve, label='Average', linewidth=2, color='#0000CD', alpha=0.8)  # Plot the average curve with higher transparency
        plt.plot(average_curve2, label=f'Standard Average', linewidth=2, color='#FF0000', alpha=0.8)  # Plot the average curve with higher transparency
        
        plt.legend()
        
        plt.xlabel('Step')
        if key == "Horizon Distance" or key == "Vertical Distance":
            plt.ylabel(key + '(m)')
        elif key == "Speed":
            plt.ylabel(key + '(m/s)')
        else:
            plt.ylabel(key)
        # plt.title(f'{key}')

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/test_height.png')  # Save the combined plot to a single file
    plt.close()
    print("Complete!")
    
    
def test_groundVer2():
    # print("??????????")
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_ground') 
    ea.Reload()
    # print(ea.scalars.Keys())
    # print("??????????")
    batch_size = 8
    
    data = defaultdict(list)
    
    # Collect data for each type
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Speed', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss']:
            data_values = [j.value for j in ea.scalars.Items(f'{data_name}{i}')]
            data[data_name].append(data_values)
    
    ea2 = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_main') 
    ea2.Reload()
    
    data2 = defaultdict(list)
    
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Speed', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss']:
            data_values = [j.value for j in ea2.scalars.Items(f'{data_name}{i}')]
            data2[data_name].append(data_values)
    
    # Plot each type of data separately
    plt.figure(figsize=(10, 20))  # Set the figure size wider and taller based on the number of data types
    for idx, (key, val) in enumerate(data.items()):
        plt.subplot(len(data), 1, idx + 1)  # Vertical arrangement with each data type in a separate row
        average_curve = np.zeros(len(val[0]))  # Initialize an array to store the average curve
        for batch_data in val:
            plt.plot(batch_data, alpha=0.3)  # Original data with moderate transparency
            average_curve += np.array(batch_data)
            
        val2 = data2[key]
        for i, batch_data2 in enumerate(val2):
            if i == 0:
                average_curve2 = np.array(batch_data2)
            else:
                average_curve2 += np.array(batch_data)
        average_curve /= len(val)  # Calculate the average curve
        average_curve2 /= len(val)
        plt.plot(average_curve, label='Average', linewidth=2, color='#0000CD', alpha=0.8)  # Plot the average curve with higher transparency
        plt.plot(average_curve2, label=f'Standard Average', linewidth=2, color='#FF0000', alpha=0.8)  # Plot the average curve with higher transparency
        
        plt.legend()
        
        plt.xlabel('Step')
        if key == "Horizon Distance" or key == "Vertical Distance":
            plt.ylabel(key + '(m)')
        elif key == "Speed":
            plt.ylabel(key + '(m/s)')
        else:
            plt.ylabel(key)
        # plt.title(f'{key}')

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/test_ground.png')  # Save the combined plot to a single file
    plt.close()
    print("Complete!")
    

def main_test():
    # print("??????????")
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/test_main__h') 
    ea.Reload()
    # print(ea.scalars.Keys())
    # print("??????????")
    batch_size = 8
    
    data = defaultdict(list)
    
    # Collect data for each type
    for i in range(batch_size):
        for data_name in ['Horizon Distance', 'Vertical Distance', 'Speed', 'Height', 'Total Loss', 'Direction Loss', 'Speed Loss', 'Orientation Loss']:
            data_values = [j.value for j in ea.scalars.Items(f'{data_name}{i}')]
            data[data_name].append(data_values)
    
    # Plot each type of data separately
    plt.figure(figsize=(10, 20))  # Set the figure size wider and taller based on the number of data types
    for idx, (key, val) in enumerate(data.items()):
        plt.subplot(len(data), 1, idx + 1)  # Vertical arrangement with each data type in a separate row
        average_curve = np.zeros(len(val[0]))  # Initialize an array to store the average curve
        for batch_data in val:
            plt.plot(batch_data, alpha=0.3)  # Original data with moderate transparency
            average_curve += np.array(batch_data)
        average_curve /= len(val)  # Calculate the average curve
        plt.plot(average_curve, label='Average', linewidth=2, color='#0000CD', alpha=0.8)  # Plot the average curve with higher transparency

        plt.legend()
        
        plt.xlabel('Step')
        if key == "Horizon Distance" or key == "Vertical Distance":
            plt.ylabel(key + '(m)')
        elif key == "Speed":
            plt.ylabel(key + '(m/s)')
        else:
            plt.ylabel(key)
        # plt.title(f'{key}')

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/test_main.png')  # Save the combined plot to a single file
    plt.close()
    print("Complete!")
    

        
def main_trainVer2():
    #加载日志数据
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/train_nopretrain') 
    ea.Reload()
    print(ea.scalars.Keys())

    # Get scalar data
    num_reset = ea.scalars.Items("Number Reset")
    average_loss = ea.scalars.Items('Ave Loss')
    average_hor_dis = ea.scalars.Items('Val Average Distance')
    

    # Plotting
    plt.figure(figsize=(10, 10))  # Set a wider figure
    plt.subplot(3, 1, 1)  # Vertical arrangement with 3 rows, 1 column, plot 1
    plt.plot([i.step for i in average_loss], [i.value for i in average_loss], label='Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('(a) Average Loss')
    plt.legend()

    plt.subplot(3, 1, 2)  # Vertical arrangement with 3 rows, 1 column, plot 2
    plt.plot([i.step for i in average_hor_dis], [i.value for i in average_hor_dis], label='Average Horizontal Distance')
    plt.xlabel('Epoch')
    plt.ylabel('(b) Average Horizontal Distance')
    plt.legend()

    plt.subplot(3, 1, 3)  # Vertical arrangement with 3 rows, 1 column, plot 3
    plt.plot([i.step for i in num_reset], [i.value for i in num_reset], label='Reset Number')
    plt.xlabel('Epoch')
    plt.ylabel('(c) Reset Number')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/main_train.png')  # Save the plot
    plt.show()
    print("Complete!")
    
    
def main_pretrain():
    #加载日志数据
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/train_nopretrain') 
    ea.Reload()
    print(ea.scalars.Keys())
    
    ea2 = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/train_pretrain') 
    ea2.Reload()

    # Get scalar data
    num_reset = ea.scalars.Items("Number Reset")
    average_loss = ea.scalars.Items('Ave Loss')
    average_hor_dis = ea.scalars.Items('Val Average Distance')
    
    num_reset2 = ea2.scalars.Items("Number Reset")
    average_loss2 = ea2.scalars.Items('Ave Loss')
    average_hor_dis2 = ea2.scalars.Items('Val Average Distance')
    

    # Plotting
    plt.figure(figsize=(10, 10))  # Set a wider figure
    plt.subplot(3, 1, 1)  # Vertical arrangement with 3 rows, 1 column, plot 1
    plt.plot([i.step for i in average_loss], [i.value for i in average_loss], label='no pre-train', color='#FF0000', alpha=0.8)
    plt.plot([i.step for i in average_loss2], [i.value for i in average_loss2], label='pre-train', color='#0000CD', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('(a) Average Loss')
    plt.legend()

    plt.subplot(3, 1, 2)  # Vertical arrangement with 3 rows, 1 column, plot 2
    plt.plot([i.step for i in average_hor_dis], [i.value for i in average_hor_dis], label='no pre-train', color='#FF0000', alpha=0.8)
    plt.plot([i.step for i in average_hor_dis2], [i.value for i in average_hor_dis2], label='pre-train', color='#0000CD', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('(b) Average Horizontal Distance')
    plt.legend()

    plt.subplot(3, 1, 3)  # Vertical arrangement with 3 rows, 1 column, plot 3
    plt.plot([i.step for i in num_reset], [i.value for i in num_reset], label='no pre-train', color='#FF0000', alpha=0.8)
    plt.plot([i.step for i in num_reset2], [i.value for i in num_reset2], label='pre-train', color='#0000CD', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('(c) Reset Number')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/main_pretrain.png')  # Save the plot
    plt.show()
    print("Complete!")

def main_train():
    #加载日志数据
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/saved_runs/tmp_track_groundVer7__exp7__chart__42__2024-04-09 20:51:11 CST') 
    ea.Reload()
    # print(ea.scalars.Keys())

    loss_total = ea.scalars.Items('Loss')
    # loss_direction = ea.scalars.Items('Loss Direction')
    # loss_orientation = ea.scalars.Items("Loss Orientation")
    # loss_height = ea.scalars.Items("Loss Height")
    num_reset = ea.scalars.Items("Number Reset")


    x = [i.step for i in loss_total]
    y1 = [i.value for i in loss_total]
    y2 = [i.value for i in num_reset]

    fig, ax1 = plt.subplots(1, 1, figsize=(16,9), dpi=80)
    ax1.plot(x, y1, color='tab:red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y2, color='tab:blue')

    ax1.set_xlabel('Year', fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel('Personal Savings Rate', color='tab:red', fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
    ax1.grid(alpha=.4)

    ax2.set_ylabel("# Unemployed (1000's)", color='tab:blue', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # ax2.set_xticks(np.arange(0, len(x), 60))
    # ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
    ax2.set_title("Personal Savings Rate vs Unemployed: Plotting in Secondary Y Axis", fontsize=22)
    fig.tight_layout()
    plt.savefig('tmp_plot.png')
    plt.show()
    print("Complete!")

if __name__ == "__main__":
    # main_train()
    # main_test()
    # main_trainVer2()
    # main_test_movingVer2()
    # main_pretrain()
    # main_test_height()
    # test_groundVer2()
    test_heightVer2()