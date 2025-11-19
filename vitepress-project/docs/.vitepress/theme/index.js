import DefaultTheme from "vitepress/theme";
import './custom.css'
import { onMounted } from "vue";

export default {
  ...DefaultTheme,
  setup() {
    onMounted(() => {

      document.querySelectorAll("details.year-block").forEach(block => {
        block.addEventListener("click", (e) => {

          // 如果点击到了 paper-card 内 → 不触发折叠
          if (e.target.closest(".paper-card")) {
            return; // 不折叠
          }

          // 如果点击的是 summary，自然触发默认行为 → 不干预
          if (e.target.closest("summary")) {
            return; // summary 自带折叠，不需要我们处理
          }

          // 其它 year-block 内的空白区域 → 手动折叠 / 展开
          block.open = !block.open;

          // 阻止事件冒泡防止 VitePress 冲突
          e.preventDefault();
        });
      });

    });
  }
};
