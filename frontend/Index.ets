import { router } from '@kit.ArkUI';

@Entry
@Component
struct Index {
  @State currentIndex: number = 0
  private swiperController: SwiperController = new SwiperController()

  build() {
    Column() {
      Swiper(this.swiperController) {
        // 第一页
        Column() {
          Text('页面 1')
            .fontSize(24)
            .fontWeight(FontWeight.Bold)
            .margin({ top: 20 })
          
          // 这里可以添加第一页的具体内容
        }
        .width('100%')
        .height('100%')
        .backgroundColor('#F1F3F5')

        // 第二页
        Column() {
          Text('页面 2')
            .fontSize(24)
            .fontWeight(FontWeight.Bold)
            .margin({ top: 20 })
          
          // 这里可以添加第二页的具体内容
        }
        .width('100%')
        .height('100%')
        .backgroundColor('#E7F5FF')

        // 第三页
        Column() {
          Text('页面 3')
            .fontSize(24)
        .fontWeight(FontWeight.Bold)
            .margin({ top: 20 })

          // 这里可以添加第三页的具体内容
        }
        .width('100%')
        .height('100%')
        .backgroundColor('#FFECB3')
      }
      .index(this.currentIndex)
      .indicator(true) // 显示页面指示器
      .loop(true) // 循环滑动
      .duration(400) // 滑动动画时长
      .onChange((index: number) => {
        this.currentIndex = index
        })
      .width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
  }
}