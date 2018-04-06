How to run the codes
===

### HW1-1 : Deep vs Shallow
- **simulate a function:**
	  在hw1的資料夾底下跑`bash run1-1_simulate.sh`會重新訓練model並畫圖存在同個資料夾底下，因為acc loss資料沒存下來，會花時間重新訓練model再畫圖，所以有事先將圖檔存在`hw1-1/pics`之中
- **Train on actual task:**
		在hw1之資料夾底下跑`bash run_1-1_MNIST.sh`會重新訓練並畫圖，同上有將資料存在資料夾之下`hw1-1/pics`

### HW1-2 : Optimization
		這邊如果要重train想必助教也是不想重train八次或是100次
		因此有把資料都存在`hw1-2`之中
- **Visualize the optimization process:**
-		在hw1的資料夾下跑`bash run1-2-1_plot.sh`只會畫圖，不會重跑並將圖存在`hw1-2/img/`資料夾下
- **Observe gradient norm during training:**
		在hw1的資料夾下跑`bash run1-2-2_plot.sh`只會畫圖，不會重跑
- **What happens when gradient is almost zero:**
		在hw1的資料夾下跑`bash run1-2-3_plot.sh`只會畫圖，不會重跑
### HW1-3 : Generalization
- **Can network fit random labels:**
	在hw1的資料夾底下跑`bash run1-3.sh 1`
- **Number of parameters v.s. Generalization:**
	在hw1的資料夾底下跑`bash run1-3.sh 2`
- **Flatness v.s. Generalization - part1:**
	在hw1的資料夾底下跑`bash run1-3.sh 3` for different batch size
	在hw1的資料夾底下跑`bash run1-3.sh 31` for different optimizer
- **Flatness v.s. Generalization - part2:**
	在hw1的資料夾底下跑`bash run1-3.sh 4`
- **Flatness v.s. Generalization - part3:**
	在hw1的資料夾底下跑`bash run1-3.sh 5`
