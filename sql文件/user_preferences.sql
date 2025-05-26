/*
 Navicat MySQL Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 80036
 Source Host           : localhost:3306
 Source Schema         : software

 Target Server Type    : MySQL
 Target Server Version : 80036
 File Encoding         : 65001

 Date: 26/05/2025
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for user_preferences
-- ----------------------------
DROP TABLE IF EXISTS `user_preferences`;
CREATE TABLE `user_preferences`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL COMMENT '关联的用户ID',
  `common_commands` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '常用指令 (JSON格式)',
  `interaction_habits` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '交互习惯 (JSON格式)',
  `command_aliases` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '指令别名 (JSON格式)',
  `updated_at` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `user_id_unique`(`user_id`) USING BTREE COMMENT '确保每个用户只有一条偏好设置记录',
  INDEX `ix_user_preferences_id`(`id`) USING BTREE,
  CONSTRAINT `fk_user_preferences_user_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '用户个性化偏好设置表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of user_preferences (示例数据)
-- ----------------------------
-- 假设用户ID为3 (asy) 有一些偏好设置
-- INSERT INTO `user_preferences` (`user_id`, `common_commands`, `interaction_habits`, `command_aliases`) VALUES
-- (3, '{\"打开空调\": 5, \"播放音乐\": 3}', '{\"preferred_confirmation_modality\": \"voice\"}', '{\"回家\": \"导航到家庭住址\"}');

-- ----------------------------
-- 备注: 这个表对应models.py中的UserPreference类
-- common_commands: 存储用户常用指令及其频率，JSON对象格式，例如 {"打开空调": 10, "播放周杰伦的歌": 5}
-- interaction_habits: 存储用户的交互习惯，JSON对象格式，例如 {"preferred_confirmation_modality": "gesture", "call_handling": "answer"}
-- command_aliases: 存储用户自定义的指令别名，JSON对象格式，例如 {"回家": "导航到我的家", "安静点": "暂停音乐并将音量调低"}
-- updated_at: 记录的创建和更新时间会自动管理
-- ----------------------------

SET FOREIGN_KEY_CHECKS = 1;