import os
import video_matcher

#=======================#
#     事前處理的部分     #
#=======================#
#print出所有會配對到的影片  人工檢查 確保不會出錯 並回傳共有幾場配對成功
threshold=0.5

video_matcher.pair_success_numbers(threshold)

#將所有配對成功的影片 放入paried_videos/內
video_matcher.move_paried_videos(threshold)








        
