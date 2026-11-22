// Build: typst compile --root . report/main.typ report/Group25_ProgressReport.pdf
#let S = json("../results/summary.json")
#let R = json("../results/p0_random.json")
#let F = json("../results/p3_final.json")
#let Z = json("../results/model_size.json")
#let V = json("../results/figures/visualize.json")
#let G = json("../results/figures/game_snapshots.json")
#let H = json("../results/figures/recon_over_time.json")
#let C = S.config

// Số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm phân cách hàng nghìn
#let vn(x, d: 2) = str(calc.round(x, digits: d)).replace(".", ",")
#let vint(x) = {
  let n = int(calc.round(calc.abs(x)))
  let digits = str(n)
  let out = ""
  for (i, ch) in digits.clusters().enumerate() {
    if i > 0 and calc.rem(digits.len() - i, 3) == 0 { out += "." }
    out += ch
  }
  if x < 0 { "−" + out } else { out }
}
#let sci(x) = {
  let e = calc.floor(calc.log(x, base: 10))
  [#vn(x / calc.pow(10, e), d: 1) × 10#super[#str(e).replace("-", "−")]]
}
#let wins(res) = res.filter(r => r.at("return") > 0).len()
#let meanlen(res) = res.map(r => r.length).sum() / res.len()
#let curve = S.eval_curve
#let eval_at(step) = curve.find(e => e.step == step)
#let last = curve.at(curve.len() - 1)
#let rec_first = H.rows.at(0)
#let rec_last = H.rows.at(H.rows.len() - 1)

#set document(title: "Group25_ProgressReport", author: "Bùi Công Hào")
#set page(paper: "a4", margin: (x: 1.6cm, y: 1.4cm), numbering: "1/3", number-align: right)
#set text(font: "Times New Roman", size: 10pt, lang: "vi")
#set par(justify: true, leading: 0.55em, spacing: 0.7em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 11pt)
#show heading: set block(above: 0.85em, below: 0.45em)
#set table(stroke: 0.4pt, inset: 3.5pt)
#show table: set text(size: 8.5pt)
#show figure.caption: set text(size: 8.5pt)
#show figure: set block(above: 0.6em, below: 0.6em)

#align(center)[
  #text(size: 13pt, weight: "bold")[Báo cáo tiến độ: Tìm hiểu và triển khai thuật toán Dreamer trong Reinforcement Learning] \
  #v(1pt)
  Học phần IT3190 · Nhóm 25 · Bùi Công Hào, MSSV 202400042
]

= Mô tả bài toán
#grid(columns: (1fr, 6.1cm), gutter: 10pt,
  [
    Mục tiêu của đề tài là huấn luyện một agent tự học chơi Pong, trò chơi bóng bàn trên máy Atari 2600. Agent điều
    khiển vợt màu xanh bên phải và đấu với opponent là vợt màu cam bên trái do máy điều khiển. Ở mỗi step, agent chỉ
    được nhìn ảnh màn hình rồi chọn một trong sáu action, gồm đứng yên, đi lên, đi xuống và các tổ hợp với nút bấm.
    Agent nhận reward +1 mỗi khi ghi bàn và −1 mỗi khi để opponent ghi bàn. Episode kết thúc khi một bên đạt 21 bàn,
    nên return của một episode nằm trong khoảng từ −21 đến 21. Agent không được cho biết luật chơi hay vị trí của bóng
    và vợt; mọi thứ đều phải học từ pixel.

    Bài toán khó vì ba lý do. Thứ nhất, input là ảnh có hàng nghìn pixel. Thứ hai, reward thưa: phải qua nhiều step
    agent mới biết một action là tốt hay xấu. Thứ ba, số step được tương tác với môi trường bị giới hạn ở 100 nghìn,
    tương đương gần 2 giờ chơi ở tốc độ thật, theo chuẩn Atari 100k @kaiser2020simple.
  ],
  figure(image("../results/figures/pong_input.pdf", width: 100%),
    caption: [Giao diện Pong ban đầu và input của agent.]),
)

Dreamer @hafner2020dreamer @hafner2021dreamerv2 @hafner2025dreamerv3 là thuật toán model-based RL. Thay vì chỉ học từ
những lần chơi thật, Dreamer học một world model: model này nén mỗi ảnh thành một latent state và học dự đoán latent
state tiếp theo, reward và thời điểm episode kết thúc. Sau đó actor và critic được huấn luyện hoàn toàn trên các
imagined trajectory do world model sinh ra, giống như agent tự tập dượt trong đầu. Nhờ vậy, mỗi step tương tác thật
được tận dụng cho rất nhiều bước học. Trong báo cáo này, em cài đặt lại DreamerV3 bằng PyTorch với model khoảng
#vn(Z.trainable / 1e6, d: 0)M parameters.

= Phương pháp
#figure(
  {
    let bx(body, fill: rgb("#eef3fb")) = box(fill: fill, stroke: 0.5pt, inset: 4pt, radius: 2pt, text(size: 8pt, body))
    let ar = text(size: 9pt)[ $arrow.r$ ]
    let y(body) = bx(fill: rgb("#fbf3e6"), body)
    stack(dir: ttb, spacing: 6pt,
      align(left, text(size: 8pt, weight: "bold")[World model learning từ dữ liệu thật]),
      [#bx[ảnh $x_t$] #ar #bx[encoder] #ar #bx[RSSM: latent state $(h_t, z_t)$] #ar
       #bx[decoder tái tạo ảnh · reward predictor · continue predictor]],
      align(left, text(size: 8pt, weight: "bold")[Actor-critic learning trong imagination]),
      [#y[latent state thật] #ar #y[actor chọn action] #ar #y[RSSM dự đoán latent state tiếp theo] #ar
       #y[lặp lại 15 steps] #ar #y[critic ước lượng value, cập nhật actor và critic]],
    )
  },
  caption: [Hai giai đoạn trong mỗi lần cập nhật của Dreamer.],
)

World model gồm ba phần. Encoder là một CNN biến ảnh thành vector. RSSM lưu latent state gồm phần deterministic $h_t$,
được cập nhật bằng GRU từ latent state và action ở step trước, và phần stochastic $z_t$ gồm 32 biến categorical. Khi
có ảnh thật, $z_t$ được suy ra từ cả $h_t$ và ảnh, gọi là posterior; khi imagination, $z_t$ chỉ được dự đoán từ $h_t$,
gọi là prior. Các head gồm decoder tái tạo lại ảnh, reward predictor và continue predictor cho biết episode có kết
thúc hay không. Loss của world model gồm reconstruction loss, reward loss, continue loss và KL divergence giữa
posterior và prior.

Sau mỗi lần cập nhật world model, từ mỗi latent state của dữ liệu thật, actor sinh imagined trajectory dài 15 steps
bằng world model. Critic học ước lượng expected return của các trajectory này, còn actor học tăng xác suất của những
action có value cao. Gradient của actor và critic không truyền ngược vào world model.

= Môi trường và dữ liệu
Môi trường Pong lấy từ Arcade Learning Environment @bellemare2013ale và được cấu hình theo chuẩn Atari 100k: mỗi action
được lặp lại trong 4 frame liên tiếp, ảnh được thu nhỏ về #C.size × #C.size pixel màu, và đầu mỗi episode có từ 0 đến
#C.noops step đứng yên ngẫu nhiên để các episode bắt đầu khác nhau. Reward giữ nguyên giá trị gốc của trò chơi.

Bài toán không có tập dữ liệu cho sẵn. Dữ liệu là các episode do chính agent chơi trong lúc học và được lưu vào replay
buffer. Mỗi step lưu một ảnh #C.size × #C.size × 3, action đã chọn, reward nhận được và hai cờ đánh dấu step đầu và
step cuối của episode. Sau quá trình huấn luyện, replay buffer chứa #vint(S.train.agent_steps) steps thuộc
#S.train.train_episodes episode hoàn chỉnh; #vint(C.prefill) steps đầu tiên dùng random policy. Mỗi gradient update dùng
một batch gồm #C.batch sequence, mỗi sequence #C.seq steps liên tiếp lấy ngẫu nhiên từ replay buffer. Các episode dùng
để evaluation được chơi trên một môi trường riêng với seed khác và không được đưa vào replay buffer.

#figure(
  table(columns: 2, align: (left, left),
    table.header([*Setting*], [*Value*]),
    [Training steps], [#vint(C.steps) steps (#vint(C.steps * C.repeat) frames)],
    [Gradient updates], [#vint(S.train.updates) (batch #C.batch × sequence length #C.seq)],
    [Model size], [#vn(Z.trainable / 1e6, d: 1)M parameters],
    [Latent state], [$h_t$: #C.deter; $z_t$: #C.stoch × #C.classes classes],
    [Imagination horizon], [#C.horizon steps],
    [Evaluation khi train], [mỗi #vint(C.eval_every) steps, #C.eval_episodes episodes],
    [Final evaluation], [#F.episodes episodes mới],
    [Hardware, training time], [1 × NVIDIA L4, #vn(S.train.wall_seconds / 3600, d: 1) h],
  ),
  caption: [Cấu hình huấn luyện.],
)

= Kết quả thí nghiệm hiện tại
#grid(columns: (1fr, 1fr), gutter: 10pt,
  [
    #figure(
      table(columns: (auto, 1fr, auto, auto), align: (left, center, center, center),
        table.header([*Agent*], [*Mean return*], [*Min / Max*], [*Wins*]),
        [Random policy], [#vn(R.mean) ± #vn(R.std)], [#vint(R.min) / #vint(R.max)], [#wins(R.results) / #R.episodes],
        [Dreamer], [#vn(F.mean) ± #vn(F.std)], [#vint(F.min) / #vint(F.max)], [#wins(F.results) / #F.episodes],
      ),
      caption: [Final evaluation trên #F.episodes episodes mới.],
    ) <tab-final>
    #figure(image("../results/figures/losses.pdf", width: 100%),
      caption: [Loss của world model và entropy của actor.]) <fig-loss>
  ],
  [
    #figure(image("../results/figures/learning_curve.pdf", width: 100%),
      caption: [Learning curve. Vùng tô mờ là một độ lệch chuẩn giữa #C.eval_episodes evaluation episodes.]) <fig-curve>
    #figure(image("../results/figures/episode_returns.pdf", width: 100%),
      caption: [Phân bố return của #F.episodes episodes trong final evaluation.])
  ],
)

#figure(image("../results/figures/game_snapshots.pdf", width: 100%),
  caption: [Một episode của agent sau khi huấn luyện. Agent bị dẫn #G.snapshot_scores.at(1).at(0) – #G.snapshot_scores.at(1).at(1)
  ở đầu episode, sau đó gỡ dần và kết thúc với tỉ số #G.final_score_agent – #G.final_score_cpu sau #vint(G.steps) steps.])

*Agent đã học được cách chơi.* Sau #vint(C.steps) training steps, mean return tăng từ #vn(R.mean) của random policy lên
#vn(F.mean). Dreamer thắng #wins(F.results) trên #F.episodes episodes, episode tốt nhất có return #vint(F.max), trong khi
random policy không thắng episode nào. Một episode của Dreamer dài trung bình #vint(meanlen(F.results)) steps, gấp khoảng
#vint(meanlen(F.results) / meanlen(R.results)) lần so với #vint(meanlen(R.results)) steps của random policy, cho thấy
agent đã đỡ được phần lớn các đường bóng. Do chỉ huấn luyện một lần với seed #str(C.seed), độ lệch chuẩn trong
@tab-final là độ lệch chuẩn giữa các episode.

*Quá trình học.* Trong #vint(eval_at(40000).step) steps đầu, evaluation return gần như không đổi, quanh mức
#vn(eval_at(40000).mean, d: 1). @fig-loss cho thấy trong khoảng từ 25 đến 40 nghìn steps, reward loss giảm và dynamics
KL tăng, tức world model bắt đầu nắm được thời điểm ghi bàn. Ngay sau đó actor mới cải thiện, và return tăng gần như
liên tục đến cuối quá trình huấn luyện. Return vẫn đang tăng ở #vint(C.steps) steps, nghĩa là model chưa hội tụ.
Ở step #vint(last.step), evaluation trên #last.episodes episodes cho return #vn(last.mean, d: 1), nhưng final evaluation
cùng model trên #F.episodes episodes mới chỉ cho #vn(F.mean): với ít episode, return dao động mạnh vì actor chọn action
theo xác suất.

#figure(image("../results/figures/recon_over_time.pdf", width: 100%),
  caption: [Output của decoder tại các checkpoint khác nhau trên cùng một đoạn ảnh thật.])

#figure(image("../results/figures/openloop_figure.pdf", width: 100%),
  caption: [World model sau #vint(C.steps) training steps nhìn ảnh thật đến vạch đỏ, sau đó tự dự đoán 15 steps tiếp theo chỉ từ các action.])

*World model.* Ở checkpoint đầu tiên decoder chỉ sinh ra nhiễu, MSE #sci(rec_first.mse). Đến 10–20 nghìn steps,
decoder mới tái tạo được nền sân và khung bảng điểm, chưa có vợt và chữ số còn sai. Hai vợt xuất hiện ở khoảng 40 nghìn
steps, cũng là lúc return bắt đầu tăng; quả bóng và chữ số đúng chỉ rõ dần từ 60 nghìn steps, và đến
#vint(rec_last.step) steps MSE còn #sci(rec_last.mse). Khi không được nhìn ảnh, world model vẫn dự đoán đúng vị trí
hai vợt trong cả #V.openloop_steps steps, nhưng quả bóng mờ dần ở các step xa; MSE của phần tự dự đoán là
#sci(V.mse_openloop), so với #sci(V.mse_posterior) khi được nhìn ảnh thật.

= Công việc đã hoàn thành và chưa hoàn thành
Em đã cài đặt đầy đủ Dreamer gồm môi trường Atari 100k, replay buffer, encoder, RSSM, decoder, reward predictor,
continue predictor, imagination, actor và critic. Để kiểm tra tính đúng đắn, em viết unit test cho từng thành phần và
thử thuật toán trên một môi trường đơn giản có đáp án biết trước; trên môi trường này return của agent tăng từ khoảng 10
lên 19,4 trên tối đa 20. Em đã huấn luyện và đánh giá trên Pong như trình bày ở trên, kiểm tra việc dừng và tiếp tục
huấn luyện từ checkpoint, và tạo video agent chơi. Các phần chưa thực hiện gồm huấn luyện nhiều lần với seed khác nhau
và thử trên game thứ hai như Breakout.

= Các bước tiếp theo cho báo cáo cuối kì
*Câu hỏi nghiên cứu.* World model của Dreamer học representation bằng cách tái tạo lại ảnh. Cách này buộc model mô tả
cả những chi tiết không ảnh hưởng đến việc chơi như nền sân hay bảng điểm, và decoder chiếm
#vn(Z.parts.decoder / 1e6, d: 1)M trên #vn(Z.world_model / 1e6, d: 1)M parameters của world model. Báo cáo cuối kì
sẽ trả lời câu hỏi: nếu bỏ reconstruction và thay bằng một objective khác, agent học tốt hơn hay kém đi, và chi phí
tính toán thay đổi thế nào.

*Các hướng liên quan.* Một số phương pháp gần đây đã bỏ reconstruction. DreamerPro @deng2022dreamerpro học representation
bằng cách gom nhóm các ảnh đã augment trên nền DreamerV2. MuDreamer @burchi2024mudreamer xây trên DreamerV3, thay
reconstruction bằng việc dự đoán value và action, dùng BatchNorm để tránh representation collapse và đã được thử trên
Atari 100k. TD-MPC2 @hansen2024tdmpc2 học world model bằng cách dự đoán latent state tiếp theo và dùng planning khi chọn
action, nhưng chỉ dành cho continuous control. MR.Q @fujimoto2025mrq là thuật toán model-free nhưng học representation
bằng các objective dự đoán giống world model, dùng một bộ hyperparameter chung cho nhiều loại bài toán kể cả Atari và
theo tác giả thì nhanh hơn DreamerV3 nhiều. DrQ-v2 @yarats2022drqv2 không dùng world model mà dựa vào data augmentation.
Các kết quả này gợi ý rằng reconstruction có thể không cần thiết, nhưng mỗi phương pháp thay đổi nhiều thành phần cùng
lúc nên khó tách riêng vai trò của decoder.

*Thiết kế thí nghiệm.* Em chọn MuDreamer vì phương pháp này chỉ thay phần representation learning và giữ nguyên các
phần còn lại của DreamerV3, nên so sánh với bản hiện tại là công bằng nhất. Ba cấu hình được so sánh: A là Dreamer
hiện tại có reconstruction, dùng lại lần huấn luyện trong báo cáo này; B là bản dựa trên MuDreamer do em tự cài đặt;
C bỏ reconstruction mà không thêm objective nào, dùng để kiểm tra representation collapse. MuDreamer còn đổi hệ số
của KL nên khác biệt giữa A và B không hoàn toàn do decoder; điều này sẽ được nêu rõ khi phân tích. Các cấu hình dùng
cùng môi trường, cùng 100 nghìn training steps và cùng cách evaluation, mỗi cấu hình huấn luyện với ba seed 0, 1, 2 trên
cùng loại GPU. Em sẽ so sánh return theo số steps và theo thời gian huấn luyện, GPU time, bộ nhớ và số parameters, rồi
báo cáo mean và độ lệch chuẩn giữa các seed. Nếu cách tránh collapse của MuDreamer không hiệu quả,
em sẽ thử objective dự đoán latent state tiếp theo của TD-MPC2. Nếu còn thời gian, em sẽ chạy thêm Breakout và
dùng mã nguồn của MR.Q làm mốc tham chiếu về tốc độ.

Mã nguồn: #link("https://github.com/hao20167/IT3190_world-model2")

#set text(size: 8pt)
#set par(spacing: 0.45em)
#bibliography("refs.bib", title: [Tài liệu tham khảo], style: "ieee")
