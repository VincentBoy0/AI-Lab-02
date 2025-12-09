from HashiwokakeroSolver import HashiSolver
import os
import copy
def read_test(file):
    arr = []
    with open(file, 'r') as inp:
        ls = inp.read().splitlines()
        
        for line in ls:
            if not line.strip():
                continue

            raw_row = line.split(", ")
            cleaned_row = []
            for item in raw_row:
                if item.isdigit():
                    cleaned_row.append(int(item))
                else:
                    cleaned_row.append(0)
            
            arr.append(cleaned_row)
            
    return arr


def run_test_case(input_path, output_path):
    grid = read_test(input_path) 
    # print(grid)
    
    print(f"Solving input from file: {input_path}")
    
    solver = HashiSolver(grid)
    ans = solver.solve()


    with open(output_path, "w", encoding="utf-8") as f:
        if ans is None:
            msg = "No Solution Found (UNSAT)."
            print(f"Writing input to: {output_path}")
            f.write(msg)
            return

        grid_ans = copy.deepcopy(grid)
        
        for edge in ans:
            ((u, v), cnt) = edge # Unpack tuple
        
            if u[0] == v[0]:
                r = u[0]
                for c in range(u[1] + 1, v[1]):
                    grid_ans[r][c] = "-" if cnt == 1 else "="
            else:
                c = u[1]
                for r in range(u[0] + 1, v[0]):
                    grid_ans[r][c] = "|" if cnt == 1 else "$"

        
        for row in grid_ans:
            row_str = "".join(f"\"{str(x)}\", " for x in row)
            row_str = "[" + row_str + "]"
            f.write(row_str + "\n")
            
    print(f"Writing input to: {output_path}")

def run():
    input_folder = "tests\input"
    output_folder = "tests\output"

    os.makedirs(output_folder, exist_ok=True)
    files = sorted(os.listdir(input_folder))

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, "output-" + filename[6:])

        if not os.path.isfile(input_path) or not filename.endswith(".txt"):
            continue

        run_test_case(input_path, output_path)

if __name__ == "__main__":
    run()
