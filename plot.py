import matplotlib.pyplot as plt
import re

def extract_ivf2_data(section):
    recall_values = []
    throughput_values = []
    for line in section.split('\n'):
        if 'Run' in line:
            recall_match = re.search(r'Recall = (\d+\.\d+)', line)
            throughput_match = re.search(r'Throughput = (\d+\.\d+)', line)
            if recall_match and throughput_match:
                recall_values.append(float(recall_match.group(1))*100)
                throughput_values.append(float(throughput_match.group(1)))
    return recall_values, throughput_values

def extract_table_data(section):
    recall_values = []
    throughput_values = []
    for line in section.split('\n'):
        if not line.startswith('=') and not line.startswith('  Ls'):
            parts = line.strip().split()
            if len(parts) >= 6:  # Table format with QPS and Recall@10
                try:
                    throughput = float(parts[1])
                    recall = float(parts[5])
                    throughput_values.append(throughput)
                    recall_values.append(recall)
                except (ValueError, IndexError):
                    continue
    return recall_values, throughput_values

def extract_zero_redundancy_data(section):
    recall_values = []
    throughput_values = []
    for line in section.split('\n'):
        if not line.startswith('='):
            parts = line.strip().split()
            if "Graph Degree" in line or "itopk_size" in line:  # Skip header lines
                continue
            # For YFCC format (8 columns with Overall_QPS and Overall_Recall)
            if len(parts) >= 8 and parts[0].isdigit():
                try:
                    throughput = float(parts[4])  # Overall_QPS
                    recall = float(parts[7])*100  # Overall_Recall
                    throughput_values.append(throughput)
                    recall_values.append(recall)
                except (ValueError, IndexError):
                    continue
            # For SIFT format (4 columns with QPS and Recall)
            elif len(parts) == 4 and parts[0].isdigit():
                try:
                    if parts[1] == "10":  # Only get data for topk=10
                        throughput = float(parts[2].replace('e+', 'e'))  # Handle scientific notation
                        recall = float(parts[3])*100
                        throughput_values.append(throughput)
                        recall_values.append(recall)
                except (ValueError, IndexError):
                    continue
    throughput_values, recall_values = zip(*sorted(zip(throughput_values, recall_values)))
    return recall_values, throughput_values

def extract_cagra_inline_data(section):
    recall_values = []
    throughput_values = []
    for line in section.split('\n'):
        if not line.startswith('='):
            parts = line.strip().split()
            if "Graph Degree" in line or "itopk_size" in line:  # Skip header lines
                continue
            # For YFCC format (3 columns)
            if len(parts) == 3 and parts[0].isdigit():
                try:
                    throughput = float(parts[1].replace('e+', 'e'))
                    recall = float(parts[2])*100
                    throughput_values.append(throughput)
                    recall_values.append(recall)
                except (ValueError, IndexError):
                    continue
            # For SIFT format (4 columns)
            elif len(parts) == 4 and parts[0].isdigit():
                try:
                    if parts[1] == "10":  # Only get data for topk=10
                        throughput = float(parts[2].replace('e+', 'e'))
                        recall = float(parts[3])*100
                        throughput_values.append(throughput)
                        recall_values.append(recall)
                except (ValueError, IndexError):
                    continue
    return recall_values, throughput_values

def extract_zero_redundancy_filtered_data(section):
    recall_values = []
    throughput_values = []
    for line in section.split('\n'):
        if not line.startswith('='):
            parts = line.strip().split()
            # Check if line matches the data format
            if len(parts) >= 11:  # Format: Spec S_itopk D_itopk ... Overall_QPS ... Overall_R
                try:
                    if parts[0].isdigit():  # Make sure it's a data line
                        throughput = float(parts[6])  # Overall_QPS
                        recall = float(parts[10])*100  # Overall_R, convert to percentage
                        throughput_values.append(throughput)
                        recall_values.append(recall)
                except (ValueError, IndexError):
                    continue
    throughput_values, recall_values = zip(*sorted(zip(throughput_values, recall_values)))
    return recall_values, throughput_values

# Read the result file
with open('Result.txt', 'r') as f:
    content = f.read()

# Split content into YFCC and SIFT-1M sections
datasets = content.split("Synthetic dataset SIFT-1M")
yfcc_content = datasets[0]
sift_content = datasets[1]

# Split into sections by empty lines
def get_sections(content):
    sections = {}
    current_section = []
    current_name = None
    
    for line in content.split('\n'):
        if line.strip() == '':
            if current_name and current_section:
                sections[current_name] = '\n'.join(current_section)
            current_section = []
            current_name = None
        elif not current_name and line.strip():
            current_name = line.strip()
        elif current_name:
            current_section.append(line)
    return sections

yfcc_sections = get_sections(yfcc_content)
methods = {
    "IVF2": ("X-", extract_ivf2_data),
    "Filteredvamana R=96": ("o-", extract_table_data),
    "Stitchvamana R=64": ("s-", extract_table_data),
    "Zero Redundancy IVF-CAGRA": ("^-", extract_zero_redundancy_data),
    "Zero Redundancy IVF-CAGRA + IVF Filtered Search (low-specificity single label queries)": ("p-", extract_zero_redundancy_filtered_data),
    "CAGRA inline-filter": ("d-", extract_cagra_inline_data)
}
labels = {
    "IVF2": "IVF2",
    "Filteredvamana R=96": "Filteredvamana R=96",
    "Stitchvamana R=64": "Stitchvamana R=64",
    "Zero Redundancy IVF-CAGRA": "IVF-CAGRA",
    "Zero Redundancy IVF-CAGRA + IVF Filtered Search (low-specificity single label queries)": "IVF-CAGRA + IVF Filtered",
    "CAGRA inline-filter": "CAGRA inline-filter"
}
# Plot YFCC Dataset
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})

# Plot all data in the main plot (ax1)
for method, (marker, extract_func) in methods.items():
    if method in yfcc_sections:
        recall_values, throughput_values = extract_func(yfcc_sections[method])
        if recall_values:
            ax1.plot(recall_values, throughput_values, marker, label=labels[method])

ax1.set_xlabel('Recall@10 (%)')
ax1.set_ylabel('QPS')
ax1.set_title('QPS vs Recall for YFCC Dataset')
ax1.legend()
ax1.grid(True)
ax1.set_yscale("log")

# Create zoomed plot (ax2)
# Define the zoom region
zoom_xmin, zoom_xmax = 85, 95  # Adjust these values to focus on the top-right region
zoom_ymin, zoom_ymax = 1e6, 5e6  # Adjust these values based on your data

for method, (marker, extract_func) in methods.items():
    if method in yfcc_sections:
        recall_values, throughput_values = extract_func(yfcc_sections[method])
        if recall_values:
            # Only plot points within the zoom region
            mask = [(x >= zoom_xmin) and (x <= zoom_xmax) and 
                   (y >= zoom_ymin) and (y <= zoom_ymax) 
                   for x, y in zip(recall_values, throughput_values)]
            if any(mask):
                zoom_recall = [x for x, m in zip(recall_values, mask) if m]
                zoom_throughput = [y for y, m in zip(throughput_values, mask) if m]
                ax2.plot(zoom_recall, zoom_throughput, marker, label=labels[method])

ax2.set_xlabel('Recall@10 (%)')
ax2.set_title('Zoomed View')
ax2.grid(True)
ax2.set_yscale("log")
# Only show legend if it's not too crowded
if len(ax2.get_lines()) <= 3:
    ax2.legend()

# Set the zoom region limits
ax2.set_xlim(zoom_xmin, zoom_xmax)
ax2.set_ylim(zoom_ymin, zoom_ymax)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('yfcc_qps_recall.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot SIFT-1M Dataset
plt.figure(figsize=(10, 6))
sift_sections = get_sections(sift_content)

sift_methods = {
    "Filteredvamana R=96": ("o-", extract_table_data),
    "Stitchvamana R=64": ("s-", extract_table_data),
    "Seperatevamana R=16": ("^-", extract_table_data),
    "Zero Redundancy IVF-CAGRA": ("d-", extract_zero_redundancy_data),
    "CAGRA inline-filter": ("v-", extract_cagra_inline_data)
}

for method, (marker, extract_func) in sift_methods.items():
    if method in sift_sections:
        recall_values, throughput_values = extract_func(sift_sections[method])
        if recall_values:  # Only plot if we have data
            plt.plot(recall_values, throughput_values, marker, label=method)

plt.xlabel('Recall@10 (%)')
plt.ylabel('QPS')
plt.title('QPS vs Recall for SIFT-1M Dataset')
plt.legend()
plt.grid(True)
plt.yscale("log")
plt.savefig('sift1m_qps_recall.png')
plt.close()