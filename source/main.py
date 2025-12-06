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


def run():
    input_path = "tests/input/input-01.txt"
    grid = read_test(input_path) 
    
    print(f"Đang giải bài toán từ: {input_path}")
    
    # 2. Giải
    solver = HashiSolver(grid)
    ans = solver.solve()

    # 3. Chuẩn bị đường dẫn Output
    output_dir = "tests/output"
    os.makedirs(output_dir, exist_ok=True) # Tạo thư mục nếu chưa có
    output_path = os.path.join(output_dir, "output-01.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        if ans is None:
            msg = "No Solution Found (UNSAT)."
            print(msg)
            f.write(msg)
            return

        # 4. Vẽ kết quả lên Grid (Dùng deepcopy để không hỏng grid gốc)
        grid_ans = copy.deepcopy(grid)
        
        # Ghi danh sách cạnh ra file trước
        f.write("Detected Bridges:\n")
        for edge in ans:
            ((u, v), cnt) = edge # Unpack tuple
            
            # Ghi log cạnh
            f.write(f"{u} --({cnt})--> {v}\n")
            
            # Vẽ lên grid
            if u[0] == v[0]: # Cùng hàng (Ngang)
                r = u[0]
                # Lưu ý: HashiSolver đảm bảo u < v nên range này an toàn
                for c in range(u[1] + 1, v[1]):
                    grid_ans[r][c] = "-" if cnt == 1 else "="
            else: # Cùng cột (Dọc)
                c = u[1]
                for r in range(u[0] + 1, v[0]):
                    # Bạn dùng "$" cho 2 cầu dọc, tôi giữ nguyên ý bạn
                    grid_ans[r][c] = "|" if cnt == 1 else "$"

        f.write("\nFinal Map:\n")
        
        # 5. Format grid đẹp để ghi vào file
        for row in grid_ans:
            # Chuyển tất cả phần tử thành string và nối lại cho thẳng hàng
            # Dùng :^3 để canh giữa cho đẹp nếu số có 2 chữ số
            row_str = "".join(f"{str(x):^3}" for x in row)
            f.write(row_str + "\n")
            
    print(f"Đã ghi kết quả vào: {output_path}")
if __name__ == "__main__":
    run()