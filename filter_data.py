with open('requirements.txt', 'r') as file:
    lines = file.readlines()
    print(lines)

formatted_lines = []
for line in lines:
    if line.strip() != '':
        package, version = line.strip().split()
        formatted_lines.append(f"{package}=={version}")

with open('formatted_requirements.txt', 'w') as file:
    file.write('\n'.join(formatted_lines))