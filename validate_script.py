import argparse
import subprocess
import os


def run_script_with_input(script_path, input_path):
    """
    指定されたPythonスクリプトを指定された入力ファイルで実行し、その結果を返す
    """
    try:
        result = subprocess.run(
            ['python', script_path],
            input=open(input_path, 'r').read(),
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error with input {input_path}: {e}")
        return None

def check_output(actual_output, input_path):
    """
    実際の出力と期待される出力を比較する
    """
    with open(input_path, 'r') as f:
        input_lines = [line.strip() for line in f.readlines()]
    N, M, T, L_A, L_B = map(int, input_lines.pop(0).split())

    G = [[] for _ in range(N)]

    for _ in range(M):
        u, v = map(int, input_lines.pop(0).split())
        G[u].append(v)
        G[v].append(u)

    t = list(map(int, input_lines.pop(0).split()))

    P = []
    for _ in range(N):
        x, y = map(int, input_lines.pop(0).split())
        P.append((x, y))

    try:
        lines = actual_output.strip().splitlines()
        A = [int(i) for i in lines[0].split()]
        if len(A) != L_A:
            print(f"Expected length of A: {L_A}, but got {len(A)}")
            return False
        B = [-1] * L_B
        pos_from = 0
        pos_to_idx = 0
        pos_to = t[pos_to_idx]
        cur = pos_from
        success = True
        for line in lines[1:]:
            if line.startswith("m"):
                next_node = int(line.split()[1])
                if next_node in B:
                    cur = next_node
                else:
                    success = False
                    break
                if cur == pos_to:
                    pos_from = pos_to
                    if pos_to_idx < T - 1:
                        pos_to_idx += 1
                        pos_to = t[pos_to_idx]
                    else:
                        break

            elif line.startswith("s"):
                _, length, a_start, b_start = line.split()
                length = int(length)
                a_start = int(a_start)
                b_start = int(b_start)
                if a_start + length > L_A:
                    success = False
                    break
                if b_start + length > L_B:
                    success = False
                    break
                B[b_start:b_start+length] = A[a_start:a_start+length]
            else:
                return False
            
        if pos_to_idx != T - 1:
            print("Not all destinations reached")
            return False
        elif cur != pos_to:
            print("Did not reach the final destination")
            return False
            
        return success
    
    except Exception as e:
        print(f"Error comparing outputs: {e}")
        return False

def validate_script(script_path, input_dir):
    """
    スクリプトと入力を用いて、スクリプトの動作を検証する
    """
    paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    all_passed = True
    for input_path in paths[:30]:
        print(f"Running script with input: {input_path}")
        actual_output = run_script_with_input(script_path, input_path)
        if actual_output is None:
            print(f"Failed to run the script for input {input_path}. Skipping...")
            all_passed = False
            continue

        checked_output = check_output(actual_output, input_path)
        if checked_output:
            print(f"Test passed for input: {input_path}")
        else:
            print(f"Test failed for input: {input_path}")
            all_passed = False

    return all_passed

if __name__ == "__main__":

    # テストしたいPythonスクリプトと入力・出力ファイルのペアを指定
    parser = argparse.ArgumentParser()

    parser.add_argument("script_path", type=str, help="Path to the Python script to validate")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing input files")
    
    args = parser.parse_args()

    script_path = args.script_path
    input_dir = args.input_dir

    result = validate_script(script_path, input_dir)
    if result:
        print("All tests passed!")
    else:
        print("Some tests failed.")
